#!/usr/bin/env bash
#
# Download and unpack EuRoC MAV sequences into data/euroc/.
#
# The dataset was moved to the ETH Research Collection in late 2025 and is now
# published as one archive per venue instead of one per sequence, so the
# smallest unit you can fetch is a whole venue. Each venue archive holds, per
# sequence, both the ASL folder as a nested zip and a ROS 1 bag of the same
# data. Only the ASL folders are used here; set KEEP_BAGS=1 to hold on to the
# bags, which are about 2 GB each.
#
#   ./scripts/fetch_euroc.sh machine_hall vicon_room1
#
# Re-running is cheap: venues that are already unpacked are left alone.

set -euo pipefail

readonly API=https://www.research-collection.ethz.ch/server/api/core/bitstreams
readonly UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'

bitstream_id() {
  case "$1" in
    machine_hall)          echo 7b2419c1-62b5-4714-b7f8-485e5fe3e5fe ;;
    vicon_room1)           echo 02ecda9a-298f-498b-970c-b7c44334d880 ;;
    vicon_room2)           echo ea12bc01-3677-4b4c-853d-87c7870b8c44 ;;
    calibration_datasets)  echo 5732e864-10f1-49e7-befb-669ee29ff770 ;;
    *) return 1 ;;
  esac
}

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dest="$root/data/euroc"
mkdir -p "$dest"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <machine_hall|vicon_room1|vicon_room2|calibration_datasets>..." >&2
  exit 2
fi

for name in "$@"; do
  if ! id=$(bitstream_id "$name"); then
    echo "unknown archive: $name" >&2
    exit 2
  fi

  venue="$dest/$name"
  archive="$dest/$name.zip"
  echo "==> $name"

  if [[ ! -d "$venue" ]]; then
    # -C - resumes a partial file; the host supports range requests.
    curl -fL --retry 5 --retry-delay 10 -C - --no-progress-meter -A "$UA" \
         -o "$archive" "$API/$id/content"
    unzip -q -o -d "$dest" "$archive"
    [[ "${KEEP_ARCHIVES:-0}" == "1" ]] || rm -f "$archive"
  fi

  # Unpack the per-sequence ASL zips in place, so a sequence ends up as
  # data/euroc/<venue>/<sequence>/mav0/.
  while IFS= read -r -d '' inner; do
    seq_dir=$(dirname "$inner")
    if [[ ! -d "$seq_dir/mav0" ]]; then
      echo "    unpacking $(basename "$seq_dir")"
      unzip -q -o -d "$seq_dir" "$inner"
    fi
    rm -f "$inner"
  done < <(find "$venue" -mindepth 2 -maxdepth 2 -name '*.zip' -print0)

  [[ "${KEEP_BAGS:-0}" == "1" ]] || find "$venue" -name '*.bag' -delete
  find "$venue" \( -name '.DS_Store' -o -name '._*' \) -delete

  echo "==> $name done"
done

echo
printf '%-20s %8s %8s %9s\n' sequence images imu seconds
find "$dest" -maxdepth 3 -name mav0 -printf '%p\0' | sort -z | while IFS= read -r -d '' m; do
  first=$(sed -n 2p "$m/imu0/data.csv" | cut -d, -f1)
  last=$(tail -n 1 "$m/imu0/data.csv" | cut -d, -f1)
  printf '%-20s %8d %8d %9.1f\n' \
    "$(basename "$(dirname "$m")")" \
    "$(find "$m/cam0/data" -name '*.png' | wc -l)" \
    "$(( $(wc -l < "$m/imu0/data.csv") - 1 ))" \
    "$(echo "($last - $first) / 1000000000" | bc -l)"
done
