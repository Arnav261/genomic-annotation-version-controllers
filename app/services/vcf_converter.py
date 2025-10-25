"""
Real VCF file converter for genomic variants.
Parses VCF files, lifts over coordinates, and generates valid output VCF.
"""
import re
import logging
from typing import List, Dict, TextIO, Optional
from datetime import datetime
from io import StringIO

logger = logging.getLogger(__name__)

class VCFConverter:
    """
    Production-ready VCF file converter.
    Handles VCF 4.0-4.3 formats with proper validation.
    """
    
    def __init__(self, liftover_service):
        self.liftover_service = liftover_service
        self.vcf_version = "VCFv4.3"
    
    def parse_vcf_line(self, line: str) -> Optional[Dict]:
        """
        Parse a VCF variant line.
        
        VCF format: CHROM POS ID REF ALT QUAL FILTER INFO [FORMAT] [SAMPLES]
        """
        if line.startswith('#') or not line.strip():
            return None
        
        parts = line.strip().split('\t')
        
        if len(parts) < 8:
            logger.warning(f"Invalid VCF line (too few columns): {line[:50]}")
            return None
        
        try:
            return {
                "chrom": parts[0],
                "pos": int(parts[1]),
                "id": parts[2] if parts[2] != '.' else None,
                "ref": parts[3],
                "alt": parts[4],
                "qual": parts[5] if parts[5] != '.' else None,
                "filter": parts[6],
                "info": parts[7],
                "format": parts[8] if len(parts) > 8 else None,
                "samples": parts[9:] if len(parts) > 9 else []
            }
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing VCF line: {e}")
            return None
    
    def parse_vcf_header(self, vcf_content: str) -> tuple[List[str], List[Dict]]:
        """
        Parse VCF header and extract metadata.
        
        Returns:
            Tuple of (header_lines, variants)
        """
        lines = vcf_content.strip().split('\n')
        header_lines = []
        variants = []
        
        for line in lines:
            if line.startswith('##'):
                header_lines.append(line)
            elif line.startswith('#CHROM'):
                header_lines.append(line)
            else:
                variant = self.parse_vcf_line(line)
                if variant:
                    variants.append(variant)
        
        return header_lines, variants
    
    def convert_variant(
        self,
        variant: Dict,
        from_build: str,
        to_build: str
    ) -> Dict:
        """
        Convert a single variant using liftover.
        
        Returns:
            Converted variant with liftover metadata
        """
        # Liftover the position
        liftover_result = self.liftover_service.convert_coordinate(
            variant["chrom"],
            variant["pos"],
            from_build,
            to_build
        )
        
        if not liftover_result.get("success"):
            return {
                **variant,
                "liftover_status": "FAILED",
                "liftover_error": liftover_result.get("error", "Unknown error")
            }
        
        # Create converted variant
        converted = {
            "chrom": liftover_result["lifted_chrom"],
            "pos": liftover_result["lifted_pos"],
            "id": variant["id"],
            "ref": variant["ref"],
            "alt": variant["alt"],
            "qual": variant["qual"],
            "filter": variant["filter"],
            "info": self._update_info_field(
                variant["info"],
                liftover_result,
                from_build,
                to_build
            ),
            "format": variant.get("format"),
            "samples": variant.get("samples", []),
            "liftover_status": "SUCCESS",
            "liftover_confidence": liftover_result.get("confidence", 1.0),
            "original_chrom": variant["chrom"],
            "original_pos": variant["pos"]
        }
        
        return converted
    
    def _update_info_field(
        self,
        info: str,
        liftover_result: Dict,
        from_build: str,
        to_build: str
    ) -> str:
        """Add liftover information to INFO field"""
        
        # Parse existing INFO
        info_parts = []
        if info and info != '.':
            info_parts.append(info)
        
        # Add liftover metadata
        info_parts.append(f"LiftOver={from_build}_{to_build}")
        info_parts.append(f"LO_Conf={liftover_result.get('confidence', 1.0):.4f}")
        
        if liftover_result.get("ambiguous"):
            info_parts.append("LO_Ambiguous=True")
        
        return ";".join(info_parts)
    
    def convert_vcf(
        self,
        vcf_content: str,
        from_build: str = "hg19",
        to_build: str = "hg38",
        keep_failed: bool = False
    ) -> Dict:
        """
        Convert entire VCF file.
        
        Args:
            vcf_content: VCF file content as string
            from_build: Source assembly
            to_build: Target assembly
            keep_failed: Whether to include variants that failed liftover
        
        Returns:
            Dict with converted VCF content and statistics
        """
        try:
            # Parse VCF
            header_lines, variants = self.parse_vcf_header(vcf_content)
            
            logger.info(f"Converting {len(variants)} variants from {from_build} to {to_build}")
            
            # Convert each variant
            converted_variants = []
            failed_variants = []
            
            for variant in variants:
                converted = self.convert_variant(variant, from_build, to_build)
                
                if converted["liftover_status"] == "SUCCESS":
                    converted_variants.append(converted)
                else:
                    failed_variants.append(converted)
            
            # Generate new VCF content
            vcf_output = self._generate_vcf_output(
                header_lines,
                converted_variants,
                failed_variants if keep_failed else [],
                from_build,
                to_build
            )
            
            # Statistics
            stats = {
                "total_variants": len(variants),
                "converted_successfully": len(converted_variants),
                "failed_conversion": len(failed_variants),
                "success_rate": (len(converted_variants) / len(variants) * 100) if variants else 0,
                "from_build": from_build,
                "to_build": to_build
            }
            
            return {
                "vcf_content": vcf_output,
                "statistics": stats,
                "converted_variants": converted_variants,
                "failed_variants": failed_variants
            }
            
        except Exception as e:
            logger.error(f"VCF conversion failed: {e}")
            raise
    
    def _generate_vcf_output(
        self,
        original_headers: List[str],
        converted_variants: List[Dict],
        failed_variants: List[Dict],
        from_build: str,
        to_build: str
    ) -> str:
        """Generate valid VCF output file"""
        
        output = StringIO()
        
        # Write VCF version
        output.write(f"##fileformat={self.vcf_version}\n")
        
        # Write liftover metadata
        output.write(f"##source=GenomicAnnotationVersionController\n")
        output.write(f"##liftoverDate={datetime.now().strftime('%Y%m%d')}\n")
        output.write(f"##liftoverFrom={from_build}\n")
        output.write(f"##liftoverTo={to_build}\n")
        output.write(f"##convertedVariants={len(converted_variants)}\n")
        output.write(f"##failedVariants={len(failed_variants)}\n")
        
        # Write INFO field definitions
        output.write('##INFO=<ID=LiftOver,Number=1,Type=String,Description="Liftover assembly conversion">\n')
        output.write('##INFO=<ID=LO_Conf,Number=1,Type=Float,Description="Liftover confidence score">\n')
        output.write('##INFO=<ID=LO_Ambiguous,Number=0,Type=Flag,Description="Multiple possible mappings">\n')
        output.write('##INFO=<ID=LO_Failed,Number=0,Type=Flag,Description="Liftover failed">\n')
        
        # Copy other original headers (except fileformat)
        for header in original_headers:
            if not header.startswith('##fileformat') and not header.startswith('#CHROM'):
                output.write(f"{header}\n")
        
        # Write column header
        output.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO")
        
        # Add FORMAT and sample columns if present
        if converted_variants and converted_variants[0].get("format"):
            output.write(f"\tFORMAT")
            if converted_variants[0].get("samples"):
                output.write(f"\t{chr(9).join(converted_variants[0]['samples'])}")
        output.write("\n")
        
        # Write converted variants
        for variant in converted_variants:
            self._write_vcf_variant(output, variant)
        
        # Write failed variants if requested
        for variant in failed_variants:
            # Mark as failed in INFO
            variant["info"] = f"{variant.get('info', '')};LO_Failed"
            self._write_vcf_variant(output, variant, use_original_coords=True)
        
        return output.getvalue()
    
    def _write_vcf_variant(self, output: TextIO, variant: Dict, use_original_coords: bool = False):
        """Write a single variant line to VCF"""
        
        chrom = variant["original_chrom"] if use_original_coords else variant["chrom"]
        pos = variant["original_pos"] if use_original_coords else variant["pos"]
        
        line_parts = [
            chrom,
            str(pos),
            variant.get("id") or ".",
            variant["ref"],
            variant["alt"],
            variant.get("qual") or ".",
            variant.get("filter") or ".",
            variant.get("info") or "."
        ]
        
        if variant.get("format"):
            line_parts.append(variant["format"])
            if variant.get("samples"):
                line_parts.extend(variant["samples"])
        
        output.write("\t".join(line_parts) + "\n")
    
    def validate_vcf(self, vcf_content: str) -> Dict:
        """
        Validate VCF file format.
        
        Returns:
            Dict with validation results and issues found
        """
        issues = []
        warnings = []
        
        lines = vcf_content.strip().split('\n')
        
        # Check for required header
        if not any(line.startswith('##fileformat=VCF') for line in lines):
            issues.append("Missing required ##fileformat header")
        
        # Check for column header
        if not any(line.startswith('#CHROM') for line in lines):
            issues.append("Missing required #CHROM column header")
        
        # Validate variant lines
        variant_count = 0
        for i, line in enumerate(lines):
            if not line.startswith('#') and line.strip():
                variant_count += 1
                variant = self.parse_vcf_line(line)
                if not variant:
                    warnings.append(f"Line {i+1}: Could not parse variant line")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "variant_count": variant_count,
            "has_samples": any('\t' in line and len(line.split('\t')) > 8 
                              for line in lines if not line.startswith('#'))
        }