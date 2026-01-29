export CUDA_VISIBLE_DEVICES="0"
python -c "import torch; print('CUDA is available! 🚀' if torch.cuda.is_available() else 'CUDA is NOT available ❌')"

cd /nfs/primary/sas_cross_encoder/january_potential_research_questions/
python monot5_sans.py

cd /nfs/primary/sas_cross_encoder/monoBERT/
python monobert_sans.py

cd /nfs/primary/sas_cross_encoder/monoELECTRA/
python monoelectra_sans.py