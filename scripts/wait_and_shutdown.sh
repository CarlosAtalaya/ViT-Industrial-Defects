#!/bin/bash
# Espera a que termine EfficientNet y apaga el PC

PID=$(pgrep -f "train_efficientnet" | head -1)

echo "Esperando proceso EfficientNet (PID: $PID)..."

while kill -0 "$PID" 2>/dev/null; do sleep 30; done

echo "EfficientNet terminado. Apagando en 30s..."
sleep 30
shutdown now -h

