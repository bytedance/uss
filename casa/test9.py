class AA:
	def __init__(self):
		self.x = 3


a1 = AA()

print(a1.__repr__())

for _ in range(4):
	b1 = a1
	print(b1.__repr__())