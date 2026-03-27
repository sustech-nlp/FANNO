#!/bin/bash
# Refresh Azure AD token every 30 minutes for long-running synthesis jobs
# Token expires after ~1 hour, so refresh at 30min interval
while true; do
    az account get-access-token --resource https://cognitiveservices.azure.com/ --query accessToken -o tsv > /tmp/.fanno_azure_token 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "[$(date)] Token refreshed ($(wc -c < /tmp/.fanno_azure_token) bytes)"
    else
        echo "[$(date)] WARNING: Token refresh failed!"
    fi
    sleep 1800  # 30 minutes
done
