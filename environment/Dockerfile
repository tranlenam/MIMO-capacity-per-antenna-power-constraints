# hash:sha256:6d00e1d69adeb2fb357abef422235947af6e7eae4b030a70399213b6bd20e8ac
FROM registry.codeocean.com/codeocean/matlab:2022a-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MLM_LICENSE_FILE

COPY postInstall /
RUN /postInstall
