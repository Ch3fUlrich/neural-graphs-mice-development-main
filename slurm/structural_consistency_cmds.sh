python scripts/structural_consistency.py --data_dir $NGMD_DATA --save_dir $NGMD_DATA/analysis_files --sim_type "zscores" --connect_thresh 4
python scripts/structural_consistency.py --data_dir $NGMD_DATA --save_dir $NGMD_DATA/analysis_files --sim_type "sttc_percentiles" --connect_thresh 95
python scripts/structural_consistency.py --data_dir $NGMD_DATA --save_dir $NGMD_DATA/analysis_files --sim_type "sttc_percentiles" --connect_thresh 96
python scripts/structural_consistency.py --data_dir $NGMD_DATA --save_dir $NGMD_DATA/analysis_files --sim_type "sttc_percentiles" --connect_thresh 97
python scripts/structural_consistency.py --data_dir $NGMD_DATA --save_dir $NGMD_DATA/analysis_files --sim_type "sttc_percentiles" --connect_thresh 98