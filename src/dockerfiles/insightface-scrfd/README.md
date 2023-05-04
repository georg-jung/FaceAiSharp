# scrfd

Use the container created by the `Dockerfile` in this directory to convert the scrfd pretrained models to onnx.

1. Build a container using the Dockerfile
2. Run the container
    * `conda list` should return

    ```
    # packages in environment at /opt/conda/envs/openmmlab:
    #
    # Name                    Version                   Build  Channel
    _libgcc_mutex             0.1                 conda_forge    conda-forge
    _openmp_mutex             4.5                  2_kmp_llvm    conda-forge
    addict                    2.4.0                    pypi_0    pypi
    autotorch                 0.0.1                    pypi_0    pypi
    bcrypt                    4.0.1                    pypi_0    pypi
    blas                      1.0                         mkl
    c-ares                    1.18.1               h7f8727e_0
    ca-certificates           2023.01.10           h06a4308_0
    certifi                   2022.12.7        py38h06a4308_0
    cffi                      1.15.1                   pypi_0    pypi
    charset-normalizer        3.0.1                    pypi_0    pypi
    click                     8.1.3                    pypi_0    pypi
    cloudpickle               2.2.1                    pypi_0    pypi
    colorama                  0.4.6                    pypi_0    pypi
    coloredlogs               15.0.1           py38h06a4308_1
    configspace               0.4.11                   pypi_0    pypi
    cpuonly                   2.0                           0    pytorch
    cryptography              39.0.0                   pypi_0    pypi
    cycler                    0.11.0             pyhd8ed1ab_0    conda-forge
    cython                    0.29.32          py38h6a678d5_0
    dask                      2023.1.1                 pypi_0    pypi
    distributed               2023.1.1                 pypi_0    pypi
    fftw                      3.3.9                h27cfd23_1
    flit-core                 3.6.0              pyhd3eb1b0_0
    freetype                  2.12.1               h4a9f257_0
    fsspec                    2023.1.0                 pypi_0    pypi
    giflib                    5.2.1                h5eee18b_1
    gmp                       6.2.1                h295c915_3
    gmpy2                     2.1.2            py38heeb90bb_0
    heapdict                  1.0.1                    pypi_0    pypi
    humanfriendly             10.0             py38h06a4308_1
    idna                      3.4                      pypi_0    pypi
    importlib-metadata        6.0.0                    pypi_0    pypi
    intel-openmp              2021.4.0          h06a4308_3561
    jinja2                    3.1.2                    pypi_0    pypi
    jpeg                      9e                   h7f8727e_0
    kiwisolver                1.4.4            py38h6a678d5_0
    krb5                      1.19.4               h568e23c_0
    lcms2                     2.12                 h3be6417_0
    ld_impl_linux-64          2.38                 h1181459_1
    lerc                      3.0                  h295c915_0
    libcurl                   7.87.0               h91b91d3_0
    libdeflate                1.8                  h7f8727e_5
    libedit                   3.1.20221030         h5eee18b_0
    libev                     4.33                 h7f8727e_1
    libffi                    3.4.2                h6a678d5_6
    libgcc-ng                 12.2.0              h65d4601_19    conda-forge
    libgfortran-ng            11.2.0               h00389a5_1
    libgfortran5              11.2.0               h1234567_1
    libnghttp2                1.46.0               hce63b2e_0
    libpng                    1.6.37               hbc83047_0
    libprotobuf               3.20.3               he621ea3_0
    libssh2                   1.10.0               h8f2d780_0
    libstdcxx-ng              11.2.0               h1234567_1
    libtiff                   4.5.0                h6a678d5_1
    libwebp                   1.2.4                h11a3e52_0
    libwebp-base              1.2.4                h5eee18b_0
    llvm-openmp               14.0.6               h9e868ea_0
    locket                    1.0.0                    pypi_0    pypi
    lz4-c                     1.9.4                h6a678d5_0
    markdown                  3.4.1                    pypi_0    pypi
    markdown-it-py            2.1.0                    pypi_0    pypi
    markupsafe                2.1.2                    pypi_0    pypi
    matplotlib-base           3.4.3            py38hf4fb855_1    conda-forge
    mdurl                     0.1.2                    pypi_0    pypi
    mkl                       2021.4.0           h06a4308_640
    mkl-service               2.4.0            py38h7f8727e_0
    mkl_fft                   1.3.1            py38hd3c417c_0
    mkl_random                1.2.2            py38h51133e4_0
    mmcv-full                 1.3.3                    pypi_0    pypi
    mmdet                     2.7.0                     dev_0    <develop>
    model-index               0.1.11                   pypi_0    pypi
    mpc                       1.1.0                h10f8cd9_1
    mpfr                      4.0.2                hb69a4c5_1
    mpmath                    1.2.1            py38h06a4308_0
    msgpack                   1.0.4                    pypi_0    pypi
    ncurses                   6.4                  h6a678d5_0
    ninja                     1.10.2               h06a4308_5
    ninja-base                1.10.2               hd09550d_5
    nose                      1.3.7                    pypi_0    pypi
    numpy                     1.23.5           py38h14f4228_0
    numpy-base                1.23.5           py38h31eccc5_0
    onnx                      1.13.0           py38h12ddb61_0
    onnxruntime               1.12.1           py38h8de7196_0
    onnxsim                   0.4.13                   pypi_0    pypi
    opencv-python             4.7.0.68                 pypi_0    pypi
    openmim                   0.3.5                    pypi_0    pypi
    openssl                   1.1.1s               h7f8727e_0
    ordered-set               4.1.0                    pypi_0    pypi
    packaging                 23.0                     pypi_0    pypi
    pandas                    1.5.3                    pypi_0    pypi
    paramiko                  2.12.0                   pypi_0    pypi
    partd                     1.3.0                    pypi_0    pypi
    pillow                    9.3.0            py38hace64e9_1
    pip                       22.3.1           py38h06a4308_0
    protobuf                  3.20.3           py38h6a678d5_0
    psutil                    5.9.4                    pypi_0    pypi
    pycocotools               2.0.6            py38h26c90d9_1    conda-forge
    pycparser                 2.21                     pypi_0    pypi
    pygments                  2.14.0                   pypi_0    pypi
    pynacl                    1.5.0                    pypi_0    pypi
    pyparsing                 3.0.9              pyhd8ed1ab_0    conda-forge
    python                    3.8.16               h7a1cb2a_2
    python-dateutil           2.8.2              pyhd8ed1ab_0    conda-forge
    python-flatbuffers        2.0                pyhd3eb1b0_0
    python_abi                3.8                      2_cp38    conda-forge
    pytorch                   1.6.0               py3.8_cpu_0  [cpuonly]  pytorch
    pytorch-mutex             1.0                         cpu    pytorch
    pytz                      2022.7.1                 pypi_0    pypi
    pyyaml                    6.0                      pypi_0    pypi
    re2                       2022.04.01           h295c915_0
    readline                  8.2                  h5eee18b_0
    requests                  2.28.2                   pypi_0    pypi
    rich                      13.3.1                   pypi_0    pypi
    scipy                     1.9.3            py38h14f4228_0
    setuptools                65.6.3           py38h06a4308_0
    six                       1.16.0             pyhd3eb1b0_1
    sortedcontainers          2.4.0                    pypi_0    pypi
    sqlite                    3.40.1               h5082296_0
    sympy                     1.11.1           py38h06a4308_0
    tabulate                  0.9.0                    pypi_0    pypi
    tblib                     1.7.0                    pypi_0    pypi
    terminaltables            3.1.10                   pypi_0    pypi
    tk                        8.6.12               h1ccaba5_0
    toolz                     0.12.0                   pypi_0    pypi
    torchvision               0.7.0                  py38_cpu  [cpuonly]  pytorch
    tornado                   6.2              py38h0a891b7_1    conda-forge
    tqdm                      4.64.1                   pypi_0    pypi
    typing-extensions         4.4.0            py38h06a4308_0
    typing_extensions         4.4.0            py38h06a4308_0
    urllib3                   1.26.14                  pypi_0    pypi
    wheel                     0.37.1             pyhd3eb1b0_0
    xz                        5.2.10               h5eee18b_1
    yapf                      0.32.0                   pypi_0    pypi
    zict                      2.2.0                    pypi_0    pypi
    zipp                      3.12.0                   pypi_0    pypi
    zlib                      1.2.13               h5eee18b_0
    zstd                      1.5.2                ha4553b6_0
    ```

3. Attach VS Code to the container.
4. Copy a pretrained model to the container.
5. Copy a test image containing a face to `~/insightface/detection/scrfd/tests/data/t1.jpg`
6. In a shell in the container, execute

    ```bash
    conda activate openmmlab
    cd ~/insightface/detection/scrfd
    # replace path to your pretrained model and
    # specify config that corresponds to model
    python tools/scrfd2onnx.py configs/scrfd/scrfd_2.5g_bnkps.py _/scrfd_10g_bnkps_model.pth
    ```

7. Copy the generated onnx file to  `~/insightface/detection/scrfd/det.onnx`.
8. Copy a test image containing one or more faces to `~/insightface/detection/scrfd/tests/data/t3.jpg`.
9. Create a folder `~/insightface/detection/scrfd/outputs` because it will not be created by the following script.
10. Run `python tools/scrfd.py` to evaluate inference
