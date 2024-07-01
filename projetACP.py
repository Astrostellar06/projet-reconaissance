import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)  # On charge le dataset LFW avec un minimum de 70 images par personne et une taille de 0.4

# On en extrait les caractéristiques et les labels
X = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]
print("Total dataset size: %d" % X.shape[0])

# Diviser les données en un ensemble d'entraînement et un ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

# Appliquer l'ACP pour réduire la dimensionnalité
n_components = 150  # Nombre de composantes principales (hyperparamètre à optimiser)
print("Number of components: ", n_components)
pca = PCA(n_components=n_components, whiten=True).fit(X_train)

X_train_pca = pca.transform(X_train)    # Appliquer la transformation aux données d'entraînement
X_test_pca = pca.transform(X_test)
eigenfaces = pca.components_.reshape((n_components, lfw_people.images.shape[1], lfw_people.images.shape[2]))    # Récupérer les eigenfaces
clf = SVC(kernel='rbf', class_weight='balanced')
clf = clf.fit(X_train_pca, y_train)

y_pred = clf.predict(X_test_pca) # On prédit les noms des personnes sur l'ensemble de test
print(classification_report(y_test, y_pred, target_names=target_names))     # Afficher les résultats de classification

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):   # Fonction pour afficher les images
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

def title(y_pred, y_test, target_names, i):     # Fonction pour afficher le nom prédit et le nom réel
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    if pred_name == true_name:
        return "Predicted: " + pred_name + "\nReality: " + true_name + "\nCorrect"
    else:
        return "Predicted: " + pred_name + "\nReality: " + true_name + "\nIncorrect"

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, lfw_people.images.shape[1], lfw_people.images.shape[2])
plt.show()
