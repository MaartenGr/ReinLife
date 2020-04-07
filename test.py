import matplotlib.pyplot as plt

plt.figure(figsize=(5*5, 5))
x = range(10)
y = range(10,20,1)

a1 = plt.subplot(151)
a1.plot(y,x)
a1.set_aspect('equal')

a2 = plt.subplot(152)
a2.plot(x,y)
a2.set_aspect('equal')

a3 = plt.subplot(153)
a3.plot(x,y)
a3.set_aspect('equal')

a4 = plt.subplot(154)
a4.plot(x,y)
a4.set_aspect('equal')

a5 = plt.subplot(155)
x = range(20)
y = range(20,40,1)
a5.plot(x,y)
a5.set_aspect('equal')
plt.show()