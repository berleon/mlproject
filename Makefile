LOCAL_CONFIG=$(HOME)/.config/docker_ports/


docker_build_pytorch:
	docker build -t leon/pytorch docker/pytorch

docker_build_mongodb:
	docker build -t leon/mongodb docker/mongodb

docker_build_tensorboard:
	docker build -t leon/tensorboard docker/tensorboard

docker_build_all: docker_build_mongodb docker_build_tensorboard docker_build_pytorch

docker_rm_images:
	docker rmi leon/pytorch
	docker rmi leon/mongodb
	docker rmi leon/tensorboard

docker_pull_all:
	docker pull berleon/master_pytorch
	docker pull berleon/master_mongodb
	docker pull berleon/master_tensorboard

docker_push_all:
	docker push berleon/master_pytorch
	docker push berleon/master_mongodb
	docker push berleon/master_tensorboard


GPU = 1

PREFIX='/home/mi/leonsixt/'
JUPYTER_DIR=$(PREFIX)
MODEL_DIR='$(PREFIX)/models/'
TENSORBOARD_DIR='$(PREFIX)/runs/'
MONGO_DIR='$(PREFIX)/mongodb_experiments/'
DOCKER_MOUNTS= -v /home/mi/leonsixt:/home/mi/leonsixt -v /mnt/ssd:/mnt/ssd



docker_run_pytorch:
	mkdir -p $(LOCAL_CONFIG)
	# --ipc=host fix data loader
	GPU=$(GPU) ./agtl-docker \
		--name leon_pytorch_conda  \
		--privileged \
		--cap-add=ALL \
		--ipc=host \
		--detach \
		-e JUPYTER_DIR=$(JUPYTER_DIR) \
		-e MODEL_DIR=$(MODEL_DIR) \
		-e TENSORBOARD_DIR=$(TENSORBOARD_DIR) \
		$(DOCKER_MOUNTS)  \
		leon/pytorch
	docker port leon_pytorch_conda | \
		perl -n -e'/8888.*:([0-9]+)/ && print $$1' \
		> $(LOCAL_CONFIG)/jupyter_port
	echo `hostname` > $(LOCAL_CONFIG)/jupyter_host

docker_run_tensorboard:
	mkdir -p $(LOCAL_CONFIG)
	GPU='' ./agtl-docker \
		--name leon_tensorboard \
		-it \
		-e TENSORBOARD_DIR=$(TENSORBOARD_DIR) \
		--publish 60000-61000:6006 \
		$(DOCKER_MOUNTS)  \
		--detach \
		leon/tensorboard
	sleep 2
	echo `hostname` > $(LOCAL_CONFIG)/tensorboard_host
	docker port leon_tensorboard | \
		perl -n -e'/6006.*:([0-9]+)/ && print $$1' \
		 > $(LOCAL_CONFIG)/tensorboard_port

docker_run_mongodb:
	mkdir -p $(LOCAL_CONFIG)
	GPU='' ./agtl-docker \
		--jupyterport 686 \
		--name leon_mongodb \
		-it \
		--detach \
		--publish 5000-6000:5000 \
		--publish 27017-28000:27017 \
		-e MONGO_DIR=$(MONGO_DIR) \
		$(DOCKER_MOUNTS)  \
		--memory=4g \
		leon/mongodb
	echo `hostname` > $(LOCAL_CONFIG)/mongodb_host
	sleep 2
	docker port leon_mongodb | \
		perl -n -e'/27017.*:([0-9]+)/ && print $$1' \
		 > $(LOCAL_CONFIG)/mongodb_port
	docker port leon_mongodb | \
		perl -n -e'/5000.*:([0-9]+)/ && print $$1' \
		> $(LOCAL_CONFIG)/sacredboard_port
	docker inspect --format='{{.NetworkSettings.IPAddress}}' leon_mongodb \
		> $(LOCAL_CONFIG)/docker_mongodb_ip

docker_run_all: docker_run_mongodb docker_run_tensorboard docker_run_pytorch

docker_rm_all_containers:
	docker rm -f leon_pytorch_conda
	docker rm -f leon_mongodb
	docker rm -f leon_tensorboard

docker_save_all_containers:
	docker commit leon_sixt_pytorch_dont_stop     leon_sixt_pytorch_dont_stop_saved
	docker commit leon_sixt_mongodb_dont_stop     leon_sixt_mongodb_dont_stop_saved
	docker commit leon_tensorboard leon_sixt_tensorboard_dont_stop_saved
	docker save -o leon_sixt_pytorch_dont_stop.tar leon_sixt_pytorch_dont_stop_saved
	docker save -o leon_sixt_mongodb_dont_stop.tar     leon_sixt_mongodb_dont_stop_saved
	docker save -o leon_tensorboard.tar leon_sixt_tensorboard_dont_stop_saved

docker_load_images:
	docker load -i leon_sixt_pytorch_dont_stop.tar
	docker load -i leon_sixt_mongodb_dont_stop.tar
	docker load -i leon_tensorboard.tar
	docker tag leon_sixt_pytorch_dont_stop_saved berleon/master_pytorch
	docker tag leon_sixt_mongodb_dont_stop_saved berleon/master_mongodb
	docker tag leon_sixt_tensorboard_dont_stop_saved berleon/master_tensorboard


print_config:
	for f in $(LOCAL_CONFIG)/* ; do \
		echo `basename $$f`: `cat $$f`; \
	done

docker_zsh:
	docker exec -it --user lsixt leon_sixt_pytorch_dont_stop sh -c "cd /gpfs01/bethge/home/lsixt/master_code  && zsh"

connect_to_mongodb:
	sudo mongo `cat ~/.config/master_thesis/docker_mongodb_ip`/


# train models

train=./attribution/main.py
train_cifar:
	$(train) with train/cifar_attr_net.py model.beta=0000

