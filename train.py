from model import CycleGAN

# Model
cycleGan = CycleGAN()

# Wrap the models with DataParallel
# cycleGan.G_XtoY = DataParallel(cycleGan.G_XtoY)
# cycleGan.G_YtoX = DataParallel(cycleGan.G_YtoX)
# cycleGan.D_X = DataParallel(cycleGan.D_X)
# cycleGan.D_Y = DataParallel(cycleGan.D_Y)

# Move the models to the Multi-GPU
# cycleGan.G_XtoY = cycleGan.G_XtoY.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# cycleGan.G_YtoX = cycleGan.G_YtoX.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# cycleGan.D_X = cycleGan.D_X.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# cycleGan.D_Y = cycleGan.D_Y.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


# Train
losses = cycleGan.train(print_every=1)