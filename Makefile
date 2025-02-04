# Makefile
.PHONY: build run

build:
	docker build -t dialogue-xai-app .

remove:
	docker rm -f dialogue-xai-app || true

run: remove
	docker run -d --name dialogue-xai-app -p 4000:4000 dialogue-xai-app