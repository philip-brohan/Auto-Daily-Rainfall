# 2nd order consensus - submit extractions

# SmolVLM2
bash /home/users/philip.brohan/Projects/Auto-Daily-Rainfall-MO/scripts/aml_submit.sh --checkpoint Daily_rainfall_sample/outputs/checkpoints/HuggingFaceTB--SmolVLM2-2.2B-Instruct-20260611-095504/HuggingFaceTB--SmolVLM2-2.2B-Instruct --images-path documents/DailyRainfall_UK/consensus_1000/images --transcriptions-path documents/DailyRainfall_UK/consensus_1000/transcriptions_2 --total-shards 1 --batch-size 50 --extraction-registry outputs/extraction_registry.json extract 

# Granite4
bash /home/users/philip.brohan/Projects/Auto-Daily-Rainfall-MO/scripts/aml_submit.sh --checkpoint Daily_rainfall_sample/outputs/checkpoints/ibm-granite--granite-vision-4.1-4b-20260611-123400/ibm-granite--granite-vision-4.1-4b --images-path documents/DailyRainfall_UK/consensus_1000/images --transcriptions-path documents/DailyRainfall_UK/consensus_1000/transcriptions_2 --total-shards 1 --batch-size 50 --extraction-registry outputs/extraction_registry.json extract 

# Gemma3
bash /home/users/philip.brohan/Projects/Auto-Daily-Rainfall-MO/scripts/aml_submit.sh --checkpoint Daily_rainfall_sample/outputs/checkpoints/google--gemma-3-4b-it-20260611-121400/google--gemma-3-4b-it --images-path documents/DailyRainfall_UK/consensus_1000/images --transcriptions-path documents/DailyRainfall_UK/consensus_1000/transcriptions_2 --total-shards 1 --batch-size 50 --extraction-registry outputs/extraction_registry.json extract 

# Gemma4
bash /home/users/philip.brohan/Projects/Auto-Daily-Rainfall-MO/scripts/aml_submit.sh --checkpoint Daily_rainfall_sample/outputs/checkpoints/google--gemma-4-E4B-it-20260611-102927/google--gemma-4-E4B-it --images-path documents/DailyRainfall_UK/consensus_1000/images --transcriptions-path documents/DailyRainfall_UK/consensus_1000/transcriptions_2 --total-shards 1 --batch-size 50 --extraction-registry outputs/extraction_registry.json extract 

# Ministral
bash /home/users/philip.brohan/Projects/Auto-Daily-Rainfall-MO/scripts/aml_submit.sh --checkpoint Daily_rainfall_sample/outputs/checkpoints/mistralai--Mistral-Small-3.1-24B-Instruct-2503-20260611-103013/mistralai--Mistral-Small-3.1-24B-Instruct-2503 --images-path documents/DailyRainfall_UK/consensus_1000/images --transcriptions-path documents/DailyRainfall_UK/consensus_1000/transcriptions_2 --total-shards 4 --batch-size 15 --extraction-registry outputs/extraction_registry.json extract 

# When these jobs complete, we'll have 2nd-order consensus extractions for all 5 models on the 1000-image consensus subset, which we can then evaluate and use for fine-tuning.

# Download the extractions to the local system
bash scripts/aml_download.sh extractions
