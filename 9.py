import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator
class Generator(nn.Module):
    def __init__(self, noise_dim, text_embed_dim, img_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + text_embed_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, img_size * img_size * 3), nn.Tanh()
        )
        self.img_size = img_size

    def forward(self, noise, text_embedding):
        return self.model(torch.cat((noise, text_embedding), dim=1)).view(-1, 3, self.img_size, self.img_size)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_size, text_embed_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size * 3 + text_embed_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, img, text_embedding):
        return self.model(torch.cat((img.view(img.size(0), -1), text_embedding), dim=1))

# Parameters
noise_dim, text_embed_dim, img_size, batch_size, lr, num_epochs = 100, 128, 64, 32, 0.0002, 100
G = Generator(noise_dim, text_embed_dim, img_size).to(device)
D = Discriminator(img_size, text_embed_dim).to(device)
optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_D = optim.Adam(D.parameters(), lr=lr)
criterion = nn.BCELoss()

# Training loop
for epoch in range(num_epochs):
    real_imgs = torch.randn(batch_size, 3, img_size, img_size).to(device)  # Dummy real images
    real_text = torch.randn(batch_size, text_embed_dim).to(device)
    noise, fake_text = torch.randn(batch_size, noise_dim).to(device), torch.randn(batch_size, text_embed_dim).to(device)

    # Train Discriminator
    fake_imgs = G(noise, fake_text).detach()
    d_loss = criterion(D(real_imgs, real_text), torch.ones(batch_size, 1).to(device)) + \
             criterion(D(fake_imgs, fake_text), torch.zeros(batch_size, 1).to(device))
    optimizer_D.zero_grad()
    d_loss.backward()
    optimizer_D.step()

    # Train Generator
    g_loss = criterion(D(G(noise, fake_text), fake_text), torch.ones(batch_size, 1).to(device))
    optimizer_G.zero_grad()
    g_loss.backward()
    optimizer_G.step()

    # Log and save images
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}] | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}')
        save_image(G(noise, fake_text).data[:25], f'generated_{epoch + 1}.png', nrow=5, normalize=True)

# Save models
torch.save(G.state_dict(), 'generator.pth')
torch.save(D.state_dict(), 'discriminator.pth')
