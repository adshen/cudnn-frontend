#!/bin/bash

set -e

DATE_FOLDER=`echo $(date '+%Y-%m-%d')`
mkdir -p many_linux_wheels

for python_tag in cp314-cp314t cp314-cp314 cp313-cp313 cp312-cp312 cp311-cp311 cp310-cp310 cp39-cp39
do
    echo "Building for ${python_tag}"
    /opt/python/${python_tag}/bin/python -m venv ${python_tag}_env
    source ${python_tag}_env/bin/activate
    CMAKE_BUILD_PARALLEL_LEVEL=8 /opt/python/${python_tag}/bin/python -m pip wheel --no-deps . -w /wheels/${python_tag} -v
    deactivate
    repair_dir=$(mktemp -d many_linux_wheels/${python_tag}.XXXXXX)
    auditwheel repair /wheels/${python_tag}/*.whl -w ${repair_dir}/
    wheel=`find ${repair_dir} -maxdepth 1 -name "*.whl" -print -quit`
    wheel_name=`basename ${wheel}`
    mv ${wheel} many_linux_wheels/${wheel_name}
    wheel=many_linux_wheels/${wheel_name}
    if [[ $CI_COMMIT_BRANCH == "main" ]]; then
        echo "main branch" 
        curl -fsS -u "${ARTIFACTORY_USER}:${ARTIFACTORY_TOKEN}" -T "${wheel}" "https://artifactory.nvidia.com/artifactory/hw-cudnn-generic-local/CUDNN/cudnn_frontend/main/${DATE_FOLDER}/${wheel_name}"
    elif [[ $CI_COMMIT_BRANCH == "develop" ]]; then 
        echo "develop branch"
        curl -fsS -u "${ARTIFACTORY_USER}:${ARTIFACTORY_TOKEN}" -T "${wheel}" "https://artifactory.nvidia.com/artifactory/hw-cudnn-generic-local/CUDNN/cudnn_frontend/develop/latest/${wheel_name}"
    else 
       echo $CI_COMMIT_BRANCH
       echo "Not posting to artifactory"
    fi
done
