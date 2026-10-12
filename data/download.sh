#!/usr/bin/env bash
# IBM Telco Customer Churn sample (7,043 customers), from IBM's own GitHub repository.
set -euo pipefail
cd "$(dirname "$0")"
curl -sSL -o telco_churn.csv \
  https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
echo "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91  telco_churn.csv" | sha256sum -c -
