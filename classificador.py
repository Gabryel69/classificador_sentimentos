import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Dados de exemplo
data = {
    'mensagem': [
        "Amei o produto",
        "Muito ruim, não gostei",
        "É ok, nada demais",
        "Perfeito, superou minhas expectativas",
        "Horrível, não recomendo",
        "Neutro, poderia ser melhor"
    ],
    'sentimento': [
        "positivo",
        "negativo",
        "neutro",
        "positivo",
        "negativo",
        "neutro"
    ]
}

df = pd.DataFrame(data)

# Separando dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(df['mensagem'], df['sentimento'], test_size=0.3, random_state=42)

# Transformação de texto em vetores
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Treina o modelo
modelo = MultinomialNB()
modelo.fit(X_train_vec, y_train)

# Testa o modelo
y_pred = modelo.predict(X_test_vec)
print("✅ Acurácia:", accuracy_score(y_test, y_pred))

# Interface simples
while True:
    texto = input("Digite uma mensagem (ou 'sair'): ")
    if texto.lower() == "sair":
        break
    texto_vec = vectorizer.transform([texto])
    sentimento = modelo.predict(texto_vec)[0]
    print(f"Sentimento detectado: {sentimento}")