#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

MY_PYTHON="python"
# MNIST_ROTA="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 1000 --data_file mnist_rotations.pt    --cuda no  --seed 0"
# MNIST_PERM="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 1000 --data_file mnist_permutations.pt --cuda no  --seed 0"

#CIFAR-100 is not use n_layers and n_hiddens becuase models is resnet-18
#samples_per_task is shuffle index / e.g) number of data : 2500, samples_per_task 2000, 1~2000 shuffle, 2001~2500 fixed
CIFAR_100i="--n_layers 2 --n_hiddens 100 --data_path ../Dataset/CIFAR100/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 2500 --data_file disjoint_cifar100.pt           --cuda yes --seed 0"

# build datasets
cd datasets/
cd raw/

# download dataset
$MY_PYTHON raw.py

cd ..

# $MY_PYTHON mnist_permutations.py \
# 	--o mnist_permutations.pt \
# 	--seed 0 \
# 	--n_tasks 20

# $MY_PYTHON mnist_rotations.py \
# 	--o mnist_rotations.pt\
# 	--seed 0 \
# 	--min_rot 0 \
# 	--max_rot 180 \
# 	--n_tasks 20

$MY_PYTHON cifar100.py \
	--i ../../Dataset/CIFAR100/cifar100.pt \
	--o ../../Dataset/CIFAR100/disjoint_cifar100.pt \
	--seed 0 \
	--n_tasks 20	# n_tasks -> number of tasks

cd ..

# # model "single"
# $MY_PYTHON main.py $MNIST_ROTA --model single --lr 0.003
# $MY_PYTHON main.py $MNIST_PERM --model single --lr 0.03
# $MY_PYTHON main.py $CIFAR_100i --model single --lr 1.0
 
# # model "independent"
# $MY_PYTHON main.py $MNIST_ROTA --model independent --lr 0.1  --finetune yes 
# $MY_PYTHON main.py $MNIST_PERM --model independent --lr 0.03 --finetune yes 
# $MY_PYTHON main.py $CIFAR_100i --model independent --lr 0.3  --finetune yes 

# # model "multimodal"
# $MY_PYTHON main.py $MNIST_ROTA  --model multimodal --lr 0.1
# $MY_PYTHON main.py $MNIST_PERM  --model multimodal --lr 0.1

# # model "EWC"
# $MY_PYTHON main.py $MNIST_ROTA --model ewc --lr 0.01 --n_memories 1000 --memory_strength 1000
# $MY_PYTHON main.py $MNIST_PERM --model ewc --lr 0.1  --n_memories 10   --memory_strength 3
# $MY_PYTHON main.py $CIFAR_100i --model ewc --lr 1.0  --n_memories 10   --memory_strength 1

# # model "iCARL"
# $MY_PYTHON main.py $CIFAR_100i --model icarl --lr 1.0 --n_memories 1280 --memory_strength 1

# model "GEM"
# $MY_PYTHON main.py $MNIST_ROTA --model gem --lr 0.1 --n_memories 256 --memory_strength 0.5
# $MY_PYTHON main.py $MNIST_PERM --model gem --lr 0.1 --n_memories 256 --memory_strength 0.5

# n_memories -> [n_tasks, n_memories, inputs] / inputs : cifar(3x32x32) -> 3,072 -> flatten
# n_memories -> Each task has 256 images
$MY_PYTHON main.py $CIFAR_100i --model gem --lr 0.1 --n_memories 256 --memory_strength 0.5


# # plot results
# cd results/
# $MY_PYTHON plot_results.py
# cd ..
