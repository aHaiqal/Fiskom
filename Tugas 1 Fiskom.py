import numpy as np                                        # Import Numpy sebagai Library

print('\n'+10*'='+'OPERASI MATRIKS'+10*'='+'\n')

a = np.array(([1,2,3],
              [4,5,6],
              [7,4,3]))
b = np.array(([3,2,1],
              [2,3,1],
              [3,1,2]))

# Penjumlahan Manual
print('Hasil Penjumlahan Manual : ')
for x in range(0, len(a)):               
    for y in range(0, len(a[0])):
        print(a[x][y] + b[x][y], end= ' '),     # print hasil tambah dari baris dan kolom dari matriks a dengan baris dan kolom matriks b dengan menggunakan tanda "|" 
    print('')

# Penjumlahan dengan Numpy
print(f'\nHasil jumlah dengan Numpy =\n {np.add(a,b)}\n')                 

# Pengurangan Manual
print('Hasil pengurangan Manual :')
for x in range(0, len(a)):
    for y in range(0, len(a[0])):
        print(a[x][y] - b[x][y], end= ' '),         # print hasil tambah dari baris dan kolom dari matriks a dengan baris dan kolom matriks b dengan menggunakan tanda "|" sebagai pemisah
    print(' ')


# Pengurangan dengan Numpy    
print(f'\nHasil pengurangan dengan Numpy =\n {np.subtract(a,b)}')       

# Perkalian Dot
print(f'\nHasil Perkalian Dot :\n {np.dot(a,b)}')     

# Perkalian Cross
print(f'\nHasil Perkalian Cross :\n {np.cross(a,b)}')         

# Determinan matriks A
print(f'\nDeterminan a : \n {np.linalg.det(a)}')    

# Determinan matriks B
print(f'\nDeterminan b : \n {np.linalg.det(b)}')        

# Transpose matriks A
print(f'\nTranspose Matriks a : \n {np.transpose(a)}')     

# Transpose matriks B
print(f'\nTranspose Matriks b : \n {np.transpose(b)}'+2*'\n')     


"""
Tugas Eliminasi Gauss Jordan dan LU Decomposition Metode Crout
Matriks nya :
a = [3, -0.1, -0.2]
    [0.1, 7, -0.3]
    [0.3, -0.2, 10]
b = [7.85]
    [-19.0]
    [71.4]
"""

print(20*'='+'GAUSS JORDAN'+20*'='+'\n')
# Matriks yang akan digunakan diperhitungan :
A = np.array([
              [3, -0.1, -0.2],
              [0.1, 7, -0.3],
              [0.3, -0.2, 10]],
              float)
B = np.array([
              [7.85],
              [-19.0],
              [71.4]],
              float)

def GaussJordan(A,b):                                     # Membuat Fungsinya
    n = len(B)                                            # Penetapan panjnag B
    
    for k in range(0,n-1):                                # k = diagonal
        for i in range(k+1,n):                            # i = baris
            A[i,k] != 0.0                                 #jika a(i,j) tidak = 0 maka diskip, jika tdk maka lanjut ke pehitungan
            lam = A[i,k]/A[k,k]                           #koefisien lambda = a_ik / a_kk
            A[i,k:n] =A[i,k:n]-lam*A[k,k:n]               #nilai elemen a baru
            B[i] = B[i]-lam*b[k]                          # Baris matriks b = baris matriks b dikurangi lambda dikali kolom matriks b
          
    for i in reversed(range(0,n-1)):                      # Untuk i didalam range matriks yang dibalik
        for k in reversed(range(i+1,n)):                  # Untuk k didalam range matriks yang dibalik
            if A[i,k] != 0:                               # Jika matriks a tidak sama dengan 0, maka:
                lam = A[i,k] / A[k,k]                     # Lambda = baris dan kolom matriks a dibagi dengan kolom matriks a
                A[i,k:n] = A[i,k:n] - lam*A[k,k:n]        # matriks a = matriks a lama dikurang lambda dikali kolom matriks a
                B[i] = B[i] - lam*B[k]                    # baris matriks b = baris matriks b yang lama dikurangi lambda dikali kolom
    
    for i in range(0,n):                                  # Untuk i didalam range baris 0,n
        for k in range(0,n):                              # Untuk j didalam range kolom 0,n
            if(A[i,i] != 1):                              # Jika baris matriks a tidak sama dengan 1, maka:
                B[i] /= A[i,i]                            # Baris matriks b dibagi dengan baris matriks a
                A[i,i] /= A[i,i]                          # Baris matriks a dibagi dengan baris matriks a
    return B                                              # Mengembalikan nilai ke b
                
print(f'Matriks A: \n {A}')
print(f'Matriks B: \n {B}')
GaussJordan(A,B)
print(f'Hasilnya : \n {B}')



# Dekomposisi LU metode Crout
print(2*'\n'+10*'='+'Dekomposisi LU metode Crout'+10*'='+'\n')

A = [[3, 0.1, -0.2],                                      # Matriks A
     [0.1, 7, -0.3],
     [0.3, -0.2, 10]]

b = [7.85, -19, 71.4]                                     # Matriks B


# Fungsi Subtitusi Maju
def Subtitusi_Maju(L, b):
    y = np.full_like(b,0)                   # Membuat vektor Y sama ukurannya seperti vektor b   
    for k in range(len(b)):     
        y[k] = b[k]    
        for i in range(k):       
            y[k] = y[k] - (L[k, i]*y[i])       
        y[k] = y[k] / L[k, k]               # Menggunakan subtitusi maju untuk mencari niulai dari y
    return y

# Fungsi Subtitusi Mundur
def Subtitusi_Mundur(U, y):
    x = np.full_like(y,0)                   # Membuat vektor X sama ukurannya seperti vektor y   
    for k in range(len(x), 0, -1):      
      x[k-1] = (y[k-1] - np.dot(U[k-1, k:], x[k:])) / U[k-1, k-1] # Menggunakan subtitusi mundur untuk mencari nilai dari x 
    return x

# Fungsi Metode Crout
def crout(A):  
# Membuat dua matriks L dan U yang diisi dengan 0 dan berukuran sama dengan A
    L = np.matrix([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
    U = np.matrix([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
    n = len(A)
    
    # perulangan untuk mengatur urutan ke-j,j dari U ke 1
    for z in range(n):
        U[z,z] = 1             
        # Perulangan mulai dari L[j][j] untuk menyelesaikan kolom ke-j dari L
        for j in range(z,n):
            # Membuat L sementara
            L_smntr = float(A[j,z])   
            for k in range(z):     
                L_smntr -= L[j,k]*U[k,z]     
            L[j,z] = L_smntr
            
        # Perulangan mulai dari U[j][j+1] untuk menyelesaikan baris ke-j dari U
        for j in range(z+1, n):
            
            # Membuat U sementara
            U_smntr = float(A[z,j])
            for k in range(z):
                U_smntr -= L[z,k]*U[k,j]
            U[z,j] = U_smntr / L[z,z]
    
    return (L, U) # Mengembalikan nilai dari matriks L dan U yaitu matriks A yang didekomposisi menggunakan metode Crout

# Fungsi menghitung nilai
def PerhitunganAkhir(A, b, crout): 
    # Membuat matriks L dan U   
    L, U = crout(A)    
    print(f'L = \n {L}')
    print(f'U = \n {U}')

    # Memanggil Subtitusi maju dan mundur yang telah dihitung sebelumnya    
    y = Subtitusi_Maju(L,b)
    x = Subtitusi_Mundur(U,y)
    return x

# Masukan matriks yang akan dihitung
A = np.array([
              [3, -0.1, -0.2],
              [0.1, 7, -0.3],
              [0.3, -0.2, 10]],
              float)
b = np.array([
              [7.85],
              [-19.0],
              [71.4]],
              float)

print(f'Jawabannya : \n{PerhitunganAkhir(A,b, crout)}')

print(44*'='+'aHaiqal'+44*'=')