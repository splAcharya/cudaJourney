**# Matrix Multiplication M * M**
------------------------------------------------------------------------
Host Exec (Mode: 1): Simple CPU Code To Perform Matrix Multiplication
Device Exec 

Data Parameter(M:5555, N:7777, P:9999)=>MatA(M:5555,N:7777), MatB(N:7777,P:9999), MatC(M:5555, P:9999)
Device Name: NVIDIA GeForce RTX 3050 Ti Laptop GPU
Max Threads per Block: 1024
Shared memory per block: 49152 bytes
Max Threads per Multiprocessor: 1536
Max Blocks per Multiprocessor: 16
Number of SMs: 20


MatA(M:33,N:55), MatB(N:55,P:77), MatC(M:33, P:77)
Host Exec (Mode:1) Elapsed Time:0.318464 ms
Device Exec (Mode:2) Elapsed Time:3.625760
Device Exec (Mode:3) Elapsed Time:0.407424

MatA(M:99,N:77), MatB(N:77,P:99), MatC(M:99, P:99)
Host Exec (Mode:1) Elapsed Time:1.695648 ms
Device Exec (Mode:2) Elapsed Time:0.394432
Device Exec (Mode:3) Elapsed Time:0.478720

MatA(M:999,N:777), MatB(N:777,P:999), MatC(M:999, P:999)
Host Exec (Mode:1) Elapsed Time:2426.968018 ms
Device Exec (Mode:2) Elapsed Time:7.219552
Device Exec (Mode:3) Elapsed Time:5.435008

MatA(M:3333,N:5555), MatB(N:5555,P:7777), MatC(M:3333, P:7777)
Device Exec (Mode:2) Elapsed Time:797.535095
Device Exec (Mode:3) Elapsed Time:494.888580

MatA(M:9999,N:7777), MatB(N:7777,P:9999), MatC(M:9999, P:9999)
Device Exec (Mode:2) Elapsed Time:5779.643555
Device Exec (Mode:3) Elapsed Time:2359.624756

