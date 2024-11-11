# cudaJourney
Learning How to Use Cuda and GPu Programmig.
Major Source: 
    https://www.olcf.ornl.gov/cuda-training-series/
    This is a trainign series from Nvidia for Oakridge National Lab.
    I am starting my cuda journey from these training videos.


1. Matrix Sums
After few addition and multiplication execrises. I finally figured out how
to launch Nvidia Nsight Compute, which seems to be a GPU code profiling tool.
At this point the course is teaching out Global Memory access patterns,
the motivating exercise was to compare a ROW SUM based cuda kernel and
a column sum based cuda kernel.

Lets Say:
N = 3
A = (int *)malloc(sizeof(int) * N * N); //row major array
A = [0, 10, 20, 30, 40, 50, 60, 70, 80]
OR
A = 0   10  20
    30  40  50
    60  70  80

THis data is then loaded into GPU memory, will also be stored in
row major format.

Row Based Acces From Global Memory (GMEM)
----------------------------------------------------
THe result we want to perform is:
0  + 10 + 20 = 30
30 + 40 + 50 = 120
60 + 70 + 80 = 210

all thread idx are global threaids.
Cycle     C1    C2    C3
---------------------------
T0      : 0     10    60
T1      : 30    40    70
T2      : 60    70    80

Each thread is responsible for a row.

COlumn Based Access From GLobal Memory (GMEM)
----------------------------------------------------
The result we want to perfom is:
0  + 30 + 60 = 90
10 + 40 + 50 = 100
20 + 50 + 80 = 150

Cycle   C1  C2  C3
-------------------
T0:     0   30  90
T1:     10  40  100
T2:     20  50  150

Now, Lets Look at now the Results From Nsight Compute

Row Major Access



Columns Major Access


The compute throughput is 3x more for columns major.
THe Memory throughput is almost 2x faster fro column major.
THe duration is 2x faster for column major access.


Why Is this The Case ?
Remeber the GLobal Memory is in ROw Major FOrmat


Row Major Access
C1   |          |           |     
C2      |           |           |
C3           |          |           |  
A = [0, 10, 20, 30, 40, 50, 60, 70, 80]

At each cycle, when there is a request mamde to GMEM, threads are accessing 
memory some distance appart, so the access will need multiple fetches.
whereas the column major access reduces these number of fetches.

COlumn Major Access
C1   |  |    |     
C2              |   |   |
C3                          |   |   |  
A = [0, 10, 20, 30, 40, 50, 60, 70, 80]

-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------

