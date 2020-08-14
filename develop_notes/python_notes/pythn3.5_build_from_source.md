Build Python3.5 from source
================



- 安裝後, 使用pip如果出現以下error log, 然後pip install都失敗的話 : 
  - <span style="color:red"><b>WARNING: pip is configured with locations that require TLS/SSL, however the SSL module in Python is not available.</b></span>
  - 一個可能的原因, 是你make前, 沒有安裝ssl相關的package, 可以先使用以下指令安裝, 然後重新make
    > - sudo apt-get install openssl
    > - sudo apt-get install libssl-dev
    > - cd python3.5
    > - make

- 使用SNPE的snpe-tensorflow-to-dlc時, 如果出現以下error log:
  - <span style="color:red"><b>Failed to find necessary python package<br>libpython3.5m.so.1.0: cannot open shared object file: No such file or directory</b></span>
  - 一個解決方案, 是重新build python, 然後再configure時, 加入--enable-shared來build shared library
    > - cd python3.5
    > - ./configure --enable-shared
    > - sudo make install
