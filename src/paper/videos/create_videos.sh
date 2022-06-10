declare -a scen_types=("rural_left_turn_risk" "freeway_enter")
declare -a risks=(0.1 0.6)
declare -a scen_indices=(0 1 2 3 4 5)

for scen_index in "${scen_indices[@]}"
  do
  for risk in "${risks[@]}"  
    do
    for scen_type in "${scen_types[@]}"
        do
        bazel run //src/thesis/videos:videos_rcrsbg_single -- $scen_type $scen_index $risk
        done
    done
done