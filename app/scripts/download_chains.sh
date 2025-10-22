#!/usr/bin/env bash
set -euo pipefail

# Usage:
# ./scripts/download_chains.sh hg19 hg38
# will attempt to download hg19ToHg38.over.chain.gz to app/data/chains/

FROM_BUILD=${1:-}
TO_BUILD=${2:-}
CHAIN_DIR=${3:-app/data/chains}
mkdir -p "${CHAIN_DIR}"

if [ -z "$FROM_BUILD" ] || [ -z "$TO_BUILD" ]; then
  echo "Usage: $0 <from_build> <to_build> [output_dir]"
  exit 1
fi

FNAME="${FROM_BUILD}To${TO_BUILD}.over.chain.gz"
URL="https://hgdownload.soe.ucsc.edu/goldenPath/${FROM_BUILD}/liftOver/${FNAME}"

echo "Downloading ${URL} to ${CHAIN_DIR}/${FNAME} ..."
curl -fSL -o "${CHAIN_DIR}/${FNAME}" "${URL}"
echo "Downloaded to ${CHAIN_DIR}/${FNAME}"