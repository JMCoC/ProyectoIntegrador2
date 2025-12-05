import os, re, random, argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)

# -------------------------------------------------
# 1) Datos: CSV opcional o dataset sintético variado
# -------------------------------------------------
random.seed(42); np.random.seed(42)

SYN_POS = [
    "Excelente servicio", "Muy buena atención", "Me encantó el producto",
    "Rápido y confiable", "Todo llegó perfecto", "Calidad superior",
    "Lo recomiendo totalmente", "Volveré a comprar", "Precio justo y buena calidad",
    "El soporte fue amable", "Experiencia increíble", "Funcionó mejor de lo esperado",
    "Entregado a tiempo", "Muy satisfecho", "Cinco estrellas",
    "La comida estaba deliciosa", "El empaque impecable", "Súper recomendable",
    "Buen trato del personal", "Gran experiencia"
]

SYN_NEG = [
    "Pésimo servicio", "Muy mala atención", "Odio este producto",
    "Lento y poco confiable", "Llegó dañado", "Calidad terrible",
    "No lo recomiendo", "No vuelvo a comprar", "Caro y mala calidad",
    "El soporte fue grosero", "Experiencia horrible", "Peor de lo esperado",
    "Entregado tarde", "Muy decepcionado", "Una estrella",
    "La comida estaba fría", "El empaque roto", "Nada recomendable",
    "Mal trato del personal", "Mala experiencia"
]

def _variantes(frase: str) -> str:
    extras = ["", "!", "!!", " 🙂", " 😡", " de verdad", " en serio", " 10/10", " 1/10",
              " súper", " la verdad", " jamás", " nunca", " para nada", " recomendado", " fatal"]
    return frase + random.choice(extras)

def synthetic_dataset(mult: int = 8) -> pd.DataFrame:
    pos = [_variantes(p) for _ in range(mult) for p in SYN_POS]
    neg = [_variantes(n) for _ in range(mult) for n in SYN_NEG]
    textos = pos + neg
    etiquetas = [1]*len(pos) + [0]*len(neg)
    df = pd.DataFrame({"texto": textos, "etiqueta": etiquetas}).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def load_or_build_dataset(csv_path: str | None) -> pd.DataFrame:
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if not {"texto", "etiqueta"}.issubset(df.columns):
            raise ValueError("El CSV debe contener columnas 'texto' y 'etiqueta' (1 positivo, 0 negativo).")
        return df[["texto", "etiqueta"]].dropna()
    return synthetic_dataset()

# ---------------------
# 2) Preprocesamiento ES
# ---------------------
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

def limpiar_es(s: str) -> str:
    s = s.lower()
    s = URL_PATTERN.sub(" ", s)
    s = re.sub(r"\d+", " 0 ", s)  # normaliza números
    s = re.sub(r"[^a-záéíóúñü0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_vectorizer(max_features: int = 40000) -> TfidfVectorizer:
    SPANISH_STOPWORDS = [
        "a","acá","ahí","ajena","ajeno","al","algo","alguna","alguno","algunas","algunos","allá","alli","allí",
        "ambas","ambos","ante","antes","apenas","aproximadamente","aquel","aquella","aquellas","aquello","aquellos",
        "aquí","arriba","así","atrás","aun","aún","aunque","bajo","bastante","bien","cada","casi","cierta","cierto",
        "ciertas","ciertos","como","con","cual","cuales","cualquier","cualquiera","cualquieras","cuan","cuando","cuanta",
        "cuantas","cuanto","cuantos","de","debe","debido","del","demás","demasiada","demasiadas","demasiado","demasiados",
        "dentro","deprisa","desde","despacio","despues","después","detrás","día","dias","donde","dos","el","él","ella",
        "ellas","ellos","emplea","emplean","emplear","en","encima","entre","era","erais","eramos","eran","eras","eres",
        "es","esa","esas","ese","eso","esos","esta","está","estaba","estaban","estado","estados","estais","estamos",
        "estar","estará","estas","este","esto","estos","estoy","fin","fue","fuera","fueron","gran","ha","haber","habia",
        "habían","habida","habidas","habido","habidos","hace","haced","hacemos","hacen","hacer","hacerla","hacerlo","haces",
        "hacia","haciendo","han","hasta","hay","incluso","intenta","intentais","intentamos","intentan","intentar","intentas",
        "ir","jamás","junto","juntos","la","las","le","les","lo","los","luego","mal","mas","más","me","medio",
        "mientras","mio","mis","misma","mismas","mismo","mismos","muy","nada","nadie","ni","ninguna","ningunas","ninguno",
        "ningunos","no","nos","nosotras","nosotros","nuestra","nuestras","nuestro","nuestros","nunca","os","otra","otras",
        "otro","otros","para","parecer","pero","poca","pocas","poco","pocos","podemos","poder","podria","podriais","podriamos",
        "podrian","podrias","por","porque","primero","puede","pueden","puedo","qué","que","quien","quienes","quiza","quizas",
        "sabe","sabeis","sabemos","saben","saber","sabes","se","segun","ser","si","siempre","siendo","sin","sintiendo",
        "sobre","sois","sola","solamente","solas","solo","solos","somos","son","soy","su","sus","suya","suyas","suyo",
        "suyos","tal","tampoco","tanta","tantas","tanto","tantos","te","teneis","tenemos","tener","tenga","tengo","ti",
        "tiempo","tiene","tienen","toda","todas","todavia","todavía","todo","todos","tomar","trabaja","trabajan","trabajar",
        "trabajas","tras","tu","tus","tuya","tuyas","tuyo","tuyos","un","una","unas","uno","unos","usa","usais","usamos",
        "usan","usar","usas","usted","ustedes","va","vais","valor","vamos","van","varias","varios","vaya","verdad","verdadera",
        "vosotras","vosotros","voy","ya","yo"
    ]
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        stop_words=SPANISH_STOPWORDS
    )

# ----------------------------
# 3) Modelo: LogisticRegression
# ----------------------------
def train_model(X_train, y_train, C_values=(0.5, 1.0, 2.0, 4.0)) -> LogisticRegression:
    best_model = None
    best_score = -1
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for C in C_values:
        model = LogisticRegression(
            penalty="l2",
            solver="saga",
            max_iter=2000,
            class_weight="balanced",
            C=C,
            n_jobs=1,
            random_state=42
        )
        fold_scores = []
        for tr_idx, va_idx in skf.split(X_train, y_train):
            m = LogisticRegression(
                penalty="l2", solver="saga", max_iter=2000,
                class_weight="balanced", C=C, n_jobs=1, random_state=42
            )
            m.fit(X_train[tr_idx], y_train.iloc[tr_idx])
            preds = m.predict(X_train[va_idx])
            fold_scores.append(accuracy_score(y_train.iloc[va_idx], preds))
        mean_acc = float(np.mean(fold_scores))
        if mean_acc > best_score:
            best_score = mean_acc
            best_model = model.fit(X_train, y_train)
    return best_model

# -----------------
# 4) Evaluación extra
# -----------------
def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    try:
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
    except Exception:
        auc = np.nan
    print(f"\nAccuracy en test: {acc:.3f} | ROC-AUC: {auc if not np.isnan(auc) else 'N/A'}")
    print("\nReporte por clase:")
    print(classification_report(y_test, pred, digits=3))
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    print("Matriz de confusión (tabla):")
    print(pd.DataFrame(cm, index=["Real 0 (neg)", "Real 1 (pos)"], columns=["Pred 0 (neg)", "Pred 1 (pos)"]))
    # Heatmap rápido
    plt.figure(figsize=(4, 3))
    plt.imshow(cm, cmap="Blues")
    plt.title("Matriz de confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.xticks([0,1],["Neg","Pos"]) ; plt.yticks([0,1],["Neg","Pos"]) 
    for (i,j),v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.show()

# ---------------------------
# 5) Artefactos y utilidades
# ---------------------------
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
VEC_PATH = os.path.join(ARTIFACTS_DIR, "tfidf.joblib")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "modelo_lr.joblib")

def save_artifacts(vectorizer, model):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(vectorizer, VEC_PATH)
    joblib.dump(model, MODEL_PATH)
    print(f"\nArtefactos guardados en: {ARTIFACTS_DIR}")

def load_artifacts():
    vec = joblib.load(VEC_PATH)
    mdl = joblib.load(MODEL_PATH)
    return vec, mdl

# --------------
# 6) CLI sencillo
# --------------
def cli():
    parser = argparse.ArgumentParser(description="Clasificador de comentarios (ES)")
    parser.add_argument("modo", choices=["train", "predict", "eval"], help="Modo de ejecución")
    parser.add_argument("--csv", dest="csv", default=None, help="Ruta CSV opcional con columnas texto,etiqueta")
    parser.add_argument("--texto", dest="texto", default=None, help="Texto para predecir (predict)")
    parser.add_argument("--txt", dest="txt", default=None, help="Ruta a archivo .txt con una frase por línea (predict)")
    args = parser.parse_args()

    if args.modo == "train":
        df = load_or_build_dataset(args.csv)
        print("Muestras:", df.shape[0], "| Positivos:", int(df.etiqueta.sum()), "| Negativos:", int(len(df)-df.etiqueta.sum()))
        df["texto_clean"] = df["texto"].apply(limpiar_es)
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            df["texto_clean"], df["etiqueta"], test_size=0.2, random_state=42, stratify=df["etiqueta"]
        )
        mayoritaria = int(round(y_train.mean()))
        baseline = (y_test == mayoritaria).mean()
        print(f"Baseline (clase mayoritaria): {baseline:.3f}")

        vectorizer = build_vectorizer()
        X_train = vectorizer.fit_transform(X_train_text)
        X_test  = vectorizer.transform(X_test_text)
        model = train_model(X_train, y_train)
        evaluate(model, X_test, y_test)
        save_artifacts(vectorizer, model)

    elif args.modo == "predict":
        vec, mdl = load_artifacts()
        textos = []
        if args.texto:
            textos.append(args.texto)
        if args.txt and os.path.exists(args.txt):
            with open(args.txt, "r", encoding="utf-8") as f:
                textos.extend([line.strip() for line in f if line.strip()])
        if not textos:
            raise SystemExit("Proporcione --texto o --txt con contenido.")
        tx = [limpiar_es(t) for t in textos]
        Xn = vec.transform(tx)
        p = mdl.predict(Xn)
        etiquetas = ["positivo" if i==1 else "negativo" for i in p]
        for t, e in zip(textos, etiquetas):
            print(f"- {t} -> {e}")

    elif args.modo == "eval":
        if not os.path.exists(VEC_PATH) or not os.path.exists(MODEL_PATH):
            raise SystemExit("Primero ejecute en modo 'train' para crear artefactos.")
        df = load_or_build_dataset(args.csv)
        df["texto_clean"] = df["texto"].apply(limpiar_es)
        vec, mdl = load_artifacts()
        X = vec.transform(df["texto_clean"]) 
        evaluate(mdl, X, df["etiqueta"]) 

if __name__ == "__main__":
    cli()