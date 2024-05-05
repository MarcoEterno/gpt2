# GPT 2

## Defense Plan (attacco preventivo)

### Model Parameters

Available Flops : 10^17 (~3 hrs on M1 MAX GPU)

Training Data Required: 5*10^8 

Optimal Parameters Count : 20 M

### Tasks

1) costruire modello
   - import dataset
   - ritagliare dataset in pezzi di context length fissata
    tokenizer
   2) pos encoding
   2) attention
   3) mlp
2) trovare dati (HF)
3) allenare (morte)

## Soluzione
1) https://github.com/openai/gpt-2
   1) muori e copia il tokenire di bert - fede l'ha già fatto
   2) copiare da repo di fede
   3) same as above
   4) same as above
   Commento: Tutto banale
2) https://github.com/openai/gpt-2-output-dataset


## dubbi
- da dove viene l'embedder?
- dimensione context length

## TODO

- specificare che il modello deve vedere solo token passati
- project vacation (far partire il training)
- definire una loss e ottimizzatore
- 