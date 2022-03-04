# DeepLib Initial experiment code

## How to run the environment?
Step 1: Configure the volumn at GRec/dockerGRec/docker_compose.yml

Step 2: Build the image and start the container

```
cd GRec/dockerGRec/
docker-compose up -d
```

Step 3: Go to the container and run the training code
```
docker exec -it grec bash

cd /app/GRec/

python mainDLv2.py --dataset qiaoji_1_1 --alg_type ngcf --regs [1e-5] --lr 0.0001  --save_flag 1 --pretrain 0 --batch_size 4096 --epoch 100000 --verbose 1
```