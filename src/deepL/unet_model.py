import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolución => Normalización => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # padding=1 mantiene el tamaño espacial de la imagen (800x600)
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Descenso en la U: Maxpooling para comprimir + Doble Convolución"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Ascenso en la U: Expansión (Upsampling) + Concatenación (Skip Connection) + Doble Convolución"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # ConvTranspose2d duplica el tamaño espacial de la imagen
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Igualar tamaños por si hay discrepancias de redondeo (ej. resoluciones impares)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Conexión residual (Skip Connection): Unimos las características de la bajada con las de la subida
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Capa final: Reduce los canales de características al número de clases deseadas (1 para línea/fondo)"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """Arquitectura U-Net completa"""
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Bajada (Encoder): Extrae el contexto
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # Puente (Bottleneck): Máxima compresión, máxima comprensión semántica
        self.down4 = Down(512, 1024)
        
        # Subida (Decoder): Recupera la resolución espacial
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
        # Salida: Máscara final
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Al subir, pasamos las variables x4, x3, x2 y x1 a través de las "Skip Connections"
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# Bloque de prueba rápida
if __name__ == '__main__':
    # Simulamos un batch de 2 imágenes RGB de 800x600 generadas aleatoriamente
    tensor_prueba = torch.randn((2, 3, 600, 800)) 
    modelo = UNet(n_channels=3, n_classes=1)
    salida = modelo(tensor_prueba)
    
    print(f"Forma de entrada: {tensor_prueba.shape}")
    print(f"Forma de salida (Máscara): {salida.shape}")