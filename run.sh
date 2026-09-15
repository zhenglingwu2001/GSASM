python train.py --backbone psmnet --datatype ours --gpu_id 2
python train.py --backbone psmnet --datatype baseline --gpu_id 1
train.py --backbone psmnet --datatype d435 --gpu_id 0

python train.py --backbone raft --datatype ours --gpu_id 2
python train.py --backbone raft --datatype baseline --gpu_id 3
python train.py --backbone raft --datatype d435 --gpu_id 0

python train.py --backbone stereonet --datatype ours --gpu_id 2
python train.py --backbone stereonet --datatype baseline --gpu_id 2
python train.py --backbone stereonet --datatype d435 --gpu_id 0

