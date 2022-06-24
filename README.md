# body-gesture-recognition




# How to use


1 create `weights` directory under PROJECT path

2 Download weight files from below link and put them into `weights` directory.

https://drive.google.com/drive/folders/1lVpYal6RxQRrkCBKDqeMNxRWnUIaZqQM?usp=sharing



3 install required packages
```
python3
tqdm
Easydict
scikit-learn
pytorch==1.11
cudatoolkit=11
numpy
```

4 uncommnet below lines
```
line 199
    server_socket, addr = self.server_socket.accept()

line 273~276
    server_socket.send(len(msg).to_bytes(4, 'little'))
    response = server_socket.recv(1)
    server_socket.send(msg)
    response = server_socket.recv(1)

```

5 run demo_network.py


6 please refer to `clients.cs' to receive datas in other process.
