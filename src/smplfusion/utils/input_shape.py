class InputShape:
    def __init__(self, image_size):
        self.h,self.w = image_size[::-1]
        self.res = self.h * self.w
        self.res64 = self.res // 64
        self.res32 = self.res // 64 // 4
        self.res16 = self.res // 64 // 16
        self.res8 = self.res // 64 // 64
        self.shape   = [self.h, self.w]
        self.shape64 = [self.h // 8, self.w // 8]
        self.shape32 = [self.h // 16, self.w // 16]
        self.shape16 = [self.h // 32, self.w // 32]
        self.shape8  = [self.h // 64, self.w // 64]
    
    def reshape(self, x):
        assert len(x.shape) == 3
        if x.shape[1] == self.res64: return x.reshape([x.shape[0]] + self.shape64 + [x.shape[-1]])
        if x.shape[1] == self.res32: return x.reshape([x.shape[0]] + self.shape32 + [x.shape[-1]])
        if x.shape[1] == self.res16: return x.reshape([x.shape[0]] + self.shape16 + [x.shape[-1]])
        if x.shape[1] ==  self.res8: return x.reshape([x.shape[0]] + self.shape8  + [x.shape[-1]])
        raise Exception("Unknown shape")

    def get_res(self, q, device = 'cpu'):
        if q.shape[1] == self.res64: return 64
        if q.shape[1] == self.res32: return 32
        if q.shape[1] == self.res16: return 16
        if q.shape[1] == self.res8: return 8