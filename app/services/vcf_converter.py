"""
Production-ready VCF file converter.
Parses VCF content, performs liftover per-variant using provided liftover service,
preserves genotype/sample columns, and produces output VCF string plus statistics.
"""
import logging
import re
from typing import List, Dict, Tuple, Optional
from io import StringIO
from datetime import datetime

logger = logging.getLogger(__name__)


class VCFConverter:
    def __init__(self, liftover_service):
        self.liftover_service = liftover_service
        self.vcf_version = "VCFv4.3"

    def parse_vcf_header(self, vcf_content: str) -> Tuple[List[str], List[Dict], List[str]]:
        """
        Returns (meta_header_lines, variant_lines_parsed, header_cols)
        """
        lines = vcf_content.splitlines()
        meta = []
        header_cols = []
        variants = []
        for line in lines:
            if line.startswith("##"):
                meta.append(line)
            elif line.startswith("#CHROM"):
                header_cols = line.strip().lstrip("#").split("\t")
                meta.append(line)
            elif not line.strip():
                continue
            else:
                parsed = self.parse_vcf_line(line, header_cols)
                if parsed:
                    variants.append(parsed)
        return meta, variants, header_cols

    def parse_vcf_line(self, line: str, header_cols: List[str]) -> Optional[Dict]:
        if line.startswith("#") or not line.strip():
            return None
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 8:
            logger.warning("Invalid VCF line (too few columns): %s", line[:80])
            return None
        chrom = parts[0]
        try:
            pos = int(parts[1])
        except ValueError:
            logger.warning("Invalid POS in VCF: %s", parts[1])
            return None
        rec = {
            "chrom": chrom,
            "pos": pos,
            "id": parts[2] if parts[2] != "." else None,
            "ref": parts[3],
            "alt": parts[4],
            "qual": parts[5] if parts[5] != "." else None,
            "filter": parts[6],
            "info": parts[7],
            "format": parts[8] if len(parts) > 8 else None,
            "samples": parts[9:] if len(parts) > 9 else [],
            "raw_line": line
        }
        return rec

    def _update_info_field(self, info: str, liftover_result: Dict, from_build: str, to_build: str) -> str:
        parts = []
        if info and info != ".":
            parts.append(info)
        parts.append(f"LiftOver={from_build}_{to_build}")
        parts.append(f"LO_Conf={liftover_result.get('confidence', 0.0):.4f}")
        if liftover_result.get("ambiguous"):
            parts.append("LO_Ambiguous=True")
        if not liftover_result.get("success"):
            parts.append("LO_Failed=True")
        return ";".join(parts)

    def convert_variant(self, variant: Dict, from_build: str, to_build: str) -> Dict:
        liftover_result = self.liftover_service.convert_coordinate(
            variant["chrom"], variant["pos"], from_build, to_build
        )
        if not liftover_result.get("success"):
            return {
                **variant,
                "liftover_status": "FAILED",
                "liftover_error": liftover_result.get("error", "Unknown"),
                "liftover_result": liftover_result
            }

        converted = {
            "chrom": liftover_result["lifted_chrom"],
            "pos": liftover_result["lifted_pos"],
            "id": variant["id"],
            "ref": variant["ref"],
            "alt": variant["alt"],
            "qual": variant["qual"],
            "filter": variant["filter"],
            "info": self._update_info_field(variant["info"], liftover_result, from_build, to_build),
            "format": variant.get("format"),
            "samples": variant.get("samples", []),
            "liftover_status": "SUCCESS",
            "liftover_confidence": liftover_result.get("confidence", 1.0),
            "original_chrom": variant["chrom"],
            "original_pos": variant["pos"],
            "liftover_result": liftover_result
        }
        return converted

    def _build_vcf_output(self, original_headers: List[str], variants_converted: List[Dict], header_cols: List[str]) -> str:
        out = StringIO()
        # Ensure fileformat
        wrote_fileformat = any(h.startswith("##fileformat=") for h in original_headers)
        if not wrote_fileformat:
            out.write(f"##fileformat={self.vcf_version}\n")
        # Add liftover INFO headers
        out.write('##INFO=<ID=LiftOver,Number=1,Type=String,Description="Liftover assembly conversion">\n')
        out.write('##INFO=<ID=LO_Conf,Number=1,Type=Float,Description="Liftover confidence score">\n')
        out.write('##INFO=<ID=LO_Ambiguous,Number=0,Type=Flag,Description="Multiple possible mappings">\n')
        out.write('##INFO=<ID=LO_Failed,Number=0,Type=Flag,Description="Liftover failed">\n')
        # Copy any original meta lines other than fileformat
        for h in original_headers:
            if not h.startswith("##fileformat="):
                out.write(f"{h}\n")
        # Write header columns if provided
        if header_cols:
            out.write("#" + "\t".join(header_cols) + "\n")
        # Write variant lines
        for v in variants_converted:
            cols = [
                v["chrom"],
                str(v["pos"]),
                v["id"] or ".",
                v["ref"],
                v["alt"],
                v["qual"] or ".",
                v["filter"],
                v["info"] or "."
            ]
            if v.get("format"):
                cols.append(v["format"])
                cols.extend(v.get("samples", []))
            out.write("\t".join(cols) + "\n")
        return out.getvalue()

    def convert_vcf(self, vcf_content: str, from_build: str = "hg19", to_build: str = "hg38", keep_failed: bool = False) -> Dict:
        headers, variants, header_cols = self.parse_vcf_header(vcf_content)
        stats = {"total": len(variants), "converted_successfully": 0, "failed_conversion": 0, "ambiguous": 0}
        converted_variants = []
        failed = []
        for v in variants:
            c = self.convert_variant(v, from_build, to_build)
            if c.get("liftover_status") == "SUCCESS":
                stats["converted_successfully"] += 1
                if c.get("liftover_result", {}).get("ambiguous"):
                    stats["ambiguous"] += 1
                converted_variants.append(c)
            else:
                stats["failed_conversion"] += 1
                failed.append(c)
                if keep_failed:
                    converted_variants.append(c)
        vcf_out = self._build_vcf_output(headers, converted_variants, header_cols)
        return {
            "vcf_content": vcf_out,
            "statistics": stats,
            "failed_variants": failed,
            "generated_at": datetime.utcnow().isoformat()
        }

    def validate_vcf(self, vcf_content: str) -> Dict:
        # Minimal validation: header presence, numeric POS, sample columns consistent
        headers, variants, header_cols = self.parse_vcf_header(vcf_content)
        problems = []
        if not header_cols:
            problems.append("Missing #CHROM header line")
        for i, v in enumerate(variants):
            if v["pos"] <= 0:
                problems.append(f"Variant {i} has non-positive position")
            if not v["ref"] or not v["alt"]:
                problems.append(f"Variant {i} missing REF or ALT")
        return {"valid": len(problems) == 0, "problems": problems, "num_variants": len(variants)}