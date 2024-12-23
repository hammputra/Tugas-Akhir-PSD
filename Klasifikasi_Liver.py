# Import libraries
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from collections import Counter
import streamlit as st
import seaborn as sns
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('Liver.csv')
    return data

# Preprocessing
@st.cache_data
def preprocess_data(data):
    # Handling missing values
    imputer = SimpleImputer(strategy='most_frequent')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Label Encoding untuk kolom gender
    le_gender = LabelEncoder()
    data_imputed['gender'] = le_gender.fit_transform(data_imputed['gender'])

    # Konversi kolom target `is_patient` ke tipe numerik
    data_imputed['is_patient'] = pd.to_numeric(data_imputed['is_patient'], errors='coerce')

    # Splitting features and target
    target = 'is_patient'
    X = data_imputed.drop(columns=[target])
    y = data_imputed[target]

    # Normalisasi features
    scaler = MinMaxScaler()
    X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Handling imbalance dengan SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_normalized, y)

    return data_imputed, X_resampled, y_resampled

@st.cache_data
def train_model(X, y, model_type='naive_bayes'):


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'naive_bayes':
        model = GaussianNB()
    elif model_type == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=400)
    else:
        st.error(f"Model type '{model_type}' tidak dikenal. Harap pilih 'naive_bayes', 'knn', atau 'logistic_regression'.")
        return None, None, None, None

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred



# Main function for Streamlit app
def main():
    st.markdown("""
        <style>
        div.stSelectbox > div > div {
            cursor: pointer;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("Navigasi")
    options = st.sidebar.selectbox(
        "Pilih Halaman:",
        ["Tampilan Data", "Preprocessing", "Training Model", "Evaluasi Model", "Prediksi"]
    )

    # Load dataset
    data = load_data()

    # Tampilan Data
    if options == "Tampilan Data":
        st.title("Klasifikasi Penyakit Liver Mengguanakan Metode Naive Bayes")
        st.title("Tampilan Data")
        st.write("Berikut adalah data mentah yang digunakan:")
        st.dataframe(data.reset_index(drop=True), height=450)

        st.write("Informasi dataset:")
        st.write(data.describe())

        # Fungsi Agregat
        st.subheader("Fungsi Agregat")
        agg_option = st.selectbox("Pilih Fungsi Agregat:", ["Mean", "Median", "Standar Deviasi", "Maksimum", "Minimum"])
        numeric_columns = data.select_dtypes(include=['number']).columns

        if agg_option == "Mean":
            st.write(data[numeric_columns].mean())
        elif agg_option == "Median":
            st.write(data[numeric_columns].median())
        elif agg_option == "Standar Deviasi":
            st.write(data[numeric_columns].std())
        elif agg_option == "Maksimum":
            st.write(data[numeric_columns].max())
        elif agg_option == "Minimum":
            st.write(data[numeric_columns].min())

        # Korelasi Data
        st.subheader("Analisis Korelasi")
        data_preprocessed, _, _ = preprocess_data(data)
        corr_matrix = data_preprocessed.corr()

        # Tampilkan tabel korelasi
        st.write("Matriks Korelasi:")
        st.dataframe(corr_matrix)

        # Plot heatmap
        st.write("Heatmap Korelasi:")
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        st.pyplot(plt)
        plt.close()     


    # Preprocessing
    elif options == "Preprocessing":
        st.title("Preprocessing Data")
        st.write("Proses preprocessing mencakup beberapa langkah berikut:")

        st.subheader("1. Penanganan Missing Value")
        st.write("Langkah pertama adalah mengisi Missing value dengan strategi yang sesuai. Pada dataset ini, nilai kosong diisi dengan nilai yang paling sering muncul (*most frequent*).")

        # Menampilkan tabel jumlah missing values per kolom
        missing_values = data.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        missing_table = pd.DataFrame({
            'Kolom': missing_values.index,
            'Jumlah Missing Values': missing_values.values
        })

        if not missing_table.empty:
            st.write("Kolom yang memiliki nilai kosong dan jumlahnya:")
            st.table(missing_table)
        else:
            st.write("Tidak ada kolom dengan nilai kosong.")

        # Imputasi nilai kosong
        imputer = SimpleImputer(strategy='most_frequent')
        data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        st.write("Hasil setelah menangani nilai kosong:")
        st.dataframe(data_imputed.reset_index(drop=True), height=450) 

        # Label Encoding untuk kolom gender
        st.subheader("2. Encoding Kolom Kategori")
        st.write("Kolom `gender` yang bersifat kategori diubah menjadi angka dengan Label Encoding. Nilai 'Male' dikonversi menjadi 1, dan 'Female' menjadi 0.")
        le_gender = LabelEncoder()
        data_imputed['gender'] = le_gender.fit_transform(data_imputed['gender'])
        st.write("Hasil setelah encoding kolom `gender`:")
        st.dataframe(data_imputed[['gender']].head())

        # Konversi target `is_patient` ke tipe numerik
        st.subheader("3. Konversi Kolom Target")
        st.write("Kolom `is_patient` (target) dikonversi ke tipe numerik untuk memastikan data bisa diproses oleh model.")
        data_imputed['is_patient'] = pd.to_numeric(data_imputed['is_patient'], errors='coerce')
        st.write("Hasil setelah konversi kolom `is_patient`:")
        st.dataframe(data_imputed[['is_patient']].head())

        # Splitting features and target
        st.subheader("4. Pemisahan Fitur dan Target")
        target = 'is_patient'
        X = data_imputed.drop(columns=[target])
        y = data_imputed[target]
        st.write("Fitur (*features*):")
        st.dataframe(X.head())
        st.write("Target (*label*):")
        st.dataframe(y.head())

        # Normalisasi features
        st.subheader("5. Normalisasi Data")
        st.write("Seluruh fitur dinormalisasi ke rentang 0-1 menggunakan *Min-Max Scaling* agar setiap fitur memiliki bobot yang seimbang.")
        scaler = MinMaxScaler()
        X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        st.write("Hasil setelah normalisasi data:")
        st.dataframe(X_normalized.head())

        # Handling imbalance dengan SMOTE
        st.subheader("6. Penanganan Data Tidak Seimbang dengan SMOTE")
        st.write("Dataset yang tidak seimbang (jumlah kelas target yang tidak proporsional) dapat memengaruhi performa model. SMOTE digunakan untuk menyeimbangkan data.")
        st.subheader("Distribusi Kelas Sebelum SMOTE")
        before_counts = pd.DataFrame.from_dict(Counter(y), orient='index', columns=['Jumlah'])
        before_counts.index.name = 'Kelas'
        st.write("Distribusi kelas sebelum SMOTE:")
        st.table(before_counts)
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_normalized, y)
        st.write("Distribusi data setelah SMOTE:")
        st.write(pd.DataFrame.from_dict(Counter(y_resampled), orient='index', columns=['Jumlah']))

        st.write("Fitur setelah preprocessing lengkap:")
        st.dataframe(X_resampled.reset_index(drop=True), height=450)
      


    # Training Model
    elif options == "Training Model":
        st.title("Training Model")
        st.write("Pada halaman ini, Anda dapat melihat proses pembagian data dan pelatihan model.")

        # Preprocess data
        _, X_resampled, y_resampled = preprocess_data(data)

        def display_evaluation_results(model, X_test, y_test, y_pred, algorithm_name):
            """Fungsi untuk menampilkan hasil evaluasi model."""
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=["0", "1"], columns=["0", "1"])

            # Display results
            st.subheader(f"Klasifikasi {algorithm_name}:")
            st.write(f"**Akurasi:** {accuracy:.2f}")

            # Display evaluation metrics
            st.write("**Hasil Evaluasi:")
            st.table(pd.DataFrame(report).transpose())

            # Display confusion matrix
            st.write("**Confusion Matrix:**")
            st.table(cm_df)

            return {
                "Akurasi": accuracy,
                "Recall": report['1']['recall'],
                "Precision": report['1']['precision'],
                "F1 Score": report['1']['f1-score']
            }

        # Initialize results container
        evaluation_results = {}

        # Train Naive Bayes
        st.subheader("1. Naive Bayes")
        nb_model, X_test, y_test, y_pred_nb = train_model(X_resampled, y_resampled, model_type='naive_bayes')

        # Naive Bayes Calculation
        st.write("**Proses Perhitungan Naive Bayes:**")
        st.write("1. Hitung probabilitas prior untuk setiap kelas.")
        priors = y_test.value_counts(normalize=True)
        st.write(priors)

        st.write("2. Hitung mean dan variance untuk setiap fitur per kelas.")
        means = X_test.groupby(y_test).mean()
        variances = X_test.groupby(y_test).var()
        st.write("**Mean:**")
        st.table(means)
        st.write("**Variance:**")
        st.table(variances)

        st.write("3. Hitung probabilitas likelihood untuk setiap fitur berdasarkan distribusi Gaussian.")
        # Add explanation and computation steps for likelihoods if needed

        evaluation_results["Naive Bayes"] = display_evaluation_results(nb_model, X_test, y_test, y_pred_nb, "Naive Bayes")

        # Train KNN
        st.subheader("2. K-Nearest Neighbors (KNN)")
        knn_model, _, _, y_pred_knn = train_model(X_resampled, y_resampled, model_type='knn')

        # KNN Calculation
        st.write("**Proses Perhitungan KNN:**")
        st.write("1. Hitung jarak Euclidean antara sampel uji dan sampel pelatihan.")
        sample_test = X_test.iloc[0]  # Ambil contoh satu data uji
        distances = ((X_resampled - sample_test) ** 2).sum(axis=1).apply(lambda x: x ** 0.5)
        st.write("Contoh Jarak Euclidean:")
        st.write(distances.head())

        st.write("2. Pilih K tetangga terdekat berdasarkan jarak.")
        k = knn_model.get_params()['n_neighbors']
        nearest_neighbors = distances.nsmallest(k)
        st.write(f"K={k}, Tetangga Terdekat:")
        st.write(nearest_neighbors)

        st.write("3. Tentukan kelas berdasarkan suara mayoritas dari K tetangga.")
        neighbor_classes = y_resampled.loc[nearest_neighbors.index]
        majority_class = neighbor_classes.mode()[0]
        st.write(f"Kelas Mayoritas: {majority_class}")

        evaluation_results["KNN"] = display_evaluation_results(knn_model, X_test, y_test, y_pred_knn, "K-Nearest Neighbors")

        # Train Logistic Regression
        st.subheader("3. Logistic Regression")
        lr_model, _, _, y_pred_lr = train_model(X_resampled, y_resampled, model_type='logistic_regression')

    # Logistic Regression Calculation
        st.write("**Proses Perhitungan Logistic Regression:**")
        st.write("1. Hitung z = W.T * X + b (kombinasi linear).")
        weights = lr_model.coef_[0]
        intercept = lr_model.intercept_[0]
        sample_features = X_test.iloc[0]
        linear_combination = sum(sample_features * weights) + intercept

        logistic_regression_table = pd.DataFrame({
            "Fitur": sample_features.index,
            "Nilai Fitur": sample_features.values,
            "Bobot (W)": weights,
            "Produk (Fitur x W)": sample_features.values * weights
        })
        logistic_regression_table.loc[len(logistic_regression_table.index)] = ["Intercept", "-", intercept, intercept]
        logistic_regression_table["Kombinasi Linear (z)"] = logistic_regression_table["Produk (Fitur x W)"].sum()

        st.write("Tabel Perhitungan Logistic Regression:")
        st.table(logistic_regression_table)

        st.write(f"Kombinasi Linear (z): {linear_combination:.2f}")

        st.write("2. Terapkan fungsi sigmoid untuk mendapatkan probabilitas.")
        sigmoid = 1 / (1 + (2.718281828459045 ** -linear_combination))
        st.write(f"Probabilitas (Sigmoid): {sigmoid:.2f}")

        st.write("3. Prediksi kelas berdasarkan threshold (biasanya 0.5).")
        predicted_class = 1 if sigmoid >= 0.5 else 0
        st.write(f"Kelas Prediksi: {predicted_class}")

        st.write("4. Ulangi proses untuk semua data uji dan evaluasi hasil prediksi.")

        evaluation_results["Logistic Regression"] = display_evaluation_results(lr_model, X_test, y_test, y_pred_lr, "Logistic Regression")

        # Plot evaluation comparison
        st.subheader("Diagram Perbandingan Hasil Evaluasi")

        # Create DataFrame for plotting
        comparison_df = pd.DataFrame(evaluation_results).T
        st.write(comparison_df)

        # Plot bar chart
        st.bar_chart(comparison_df)

        # Customized matplotlib chart
        

        labels = comparison_df.index
        x = range(len(labels))  # the label locations
        width = 0.2  # the width of the bars

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar([p - width for p in x], comparison_df['Akurasi'], width, label='Akurasi')
        ax.bar(x, comparison_df['Recall'], width, label='Recall')
        ax.bar([p + width for p in x], comparison_df['Precision'], width, label='Precision')
        ax.bar([p + 2 * width for p in x], comparison_df['F1 Score'], width, label='F1 Score')

        # Add labels and title
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Grafik Hasil Evaluasi')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        # Display the chart
        st.pyplot(fig)


    # Evaluasi Model
    elif options == "Evaluasi Model":
        st.title("Evaluasi Model")
        
        # Preprocess and train models
        _, X_resampled, y_resampled = preprocess_data(data)
        nb_model, X_test, y_test, y_pred_nb = train_model(X_resampled, y_resampled, model_type='naive_bayes')
        knn_model, _, _, y_pred_knn = train_model(X_resampled, y_resampled, model_type='knn')
        lr_model, _, _, y_pred_lr = train_model(X_resampled, y_resampled, model_type='logistic_regression')

        # Function to display confusion matrix and evaluation metrics
        def display_confusion_matrix_and_metrics(y_true, y_pred, model_name):
            cm = confusion_matrix(y_true, y_pred)
            st.subheader(f"Confusion Matrix - {model_name}")
            plt.figure(figsize=(5, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
            plt.xlabel("Prediksi")
            plt.ylabel("Aktual")
            st.pyplot(plt)

            # Calculate and display evaluation metrics
            auc = roc_auc_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            
            st.write(f"AUC: {auc:.2f}")
            st.write(f"Precision: {precision:.2f}")
            st.write(f"Recall: {recall:.2f}")
            st.write(f"F1-Score: {f1:.2f}")

        # Display confusion matrices and metrics for each model
        display_confusion_matrix_and_metrics(y_test, y_pred_nb, "Naive Bayes")
        display_confusion_matrix_and_metrics(y_test, y_pred_knn, "KNN")
        display_confusion_matrix_and_metrics(y_test, y_pred_lr, "Logistic Regression")

        # Classification results table
        st.subheader("Perbandingan Akurasi")
        results = {
            "Model": ["Naive Bayes", "KNN", "Logistic Regression"],
            "Accuracy": [
                accuracy_score(y_test, y_pred_nb) * 10,
                accuracy_score(y_test, y_pred_knn) * 10,
                accuracy_score(y_test, y_pred_lr) * 10
            ]
        }
        results_df = pd.DataFrame(results)
        st.table(results_df)

        # Diagram batang untuk perbandingan akurasi
        st.subheader("Diagram Batang Perbandingan Akurasi")
        plt.figure(figsize=(8, 5))
        sns.barplot(x="Model", y="Accuracy", data=results_df, palette="Blues_d")
        plt.xlabel("Model")
        plt.ylabel("Akurasi (Skala 1-10)")
        plt.title("Perbandingan Akurasi Antar Model")
        st.pyplot(plt)


    # Prediksi
    elif options == "Prediksi":
        st.title("Prediksi Penyakit Liver")
        st.write("Masukkan data pasien untuk melakukan prediksi.")

        # Input data pasien
        user_data = {}
        user_data['age'] = st.number_input("Usia Pasien (age):", min_value=0, max_value=100, value=30)
        user_data['gender'] = st.selectbox("Jenis Kelamin (gender):", ['Male', 'Female'])
        user_data['tot_bilirubin'] = st.number_input("Total Bilirubin (tot_bilirubin):", min_value=0.0, max_value=100.0, value=1.0)
        user_data['direct_bilirubin'] = st.number_input("Direct Bilirubin (direct_bilirubin):", min_value=0.0, max_value=10.0, value=0.1)
        user_data['tot_proteins'] = st.number_input("Total Proteins (tot_proteins):", min_value=0.0, max_value=100.0, value=6.5)
        user_data['albumin'] = st.number_input("Albumin (albumin):", min_value=0.0, max_value=10.0, value=3.5)
        user_data['ag_ratio'] = st.number_input("A/G Ratio (ag_ratio):", min_value=0.0, max_value=10.0, value=1.1)
        user_data['sgpt'] = st.number_input("SGPT (sgpt):", min_value=0, max_value=1000, value=40)
        user_data['sgot'] = st.number_input("SGOT (sgot):", min_value=0, max_value=1000, value=50)
        user_data['alkphos'] = st.number_input("Alkaline Phosphatase (alkphos):", min_value=0, max_value=2000, value=200)

        # Encode gender input
        user_data['gender'] = 1 if user_data['gender'] == 'Male' else 0

        # Preprocess and train model
        _, X_resampled, y_resampled = preprocess_data(data)
        model, _, _, _ = train_model(X_resampled, y_resampled)

        # Prediction
        if st.button("Prediksi"):
            user_input = pd.DataFrame([user_data])

            # Gunakan scaler yang sudah di-fit sebelumnya
            scaler = MinMaxScaler()
            scaler.fit(X_resampled)  # Fit scaler pada data training
            user_input_normalized = pd.DataFrame(scaler.transform(user_input), columns=X_resampled.columns)

            # Prediksi
            prediction = model.predict(user_input_normalized)
            result = "Positif (Memiliki Penyakit Liver)" if prediction[0] == 1 else "Negatif (Tidak Memiliki Penyakit Liver)"

            st.subheader("Hasil Prediksi")
            st.write(f"Berdasarkan data yang dimasukkan, hasil prediksi adalah: **{result}**.")

        
# Run the app
if __name__ == "__main__":
    main()
