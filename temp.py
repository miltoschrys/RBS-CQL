kubectl get --all-namespaces  --output json  pods | jq '.items[] | select(.status.podIP=="10.22.19.69")' | jq .metadata.name
step 20 remove from dataset
remove 74 ste pfrom dataset
