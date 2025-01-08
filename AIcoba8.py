import streamlit as st
import pandas as pd
import joblib
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Kelas DecisionTree dan RandomForest
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, data, target, features, depth=0):
        if depth == self.max_depth or len(np.unique(data[target])) == 1:
            most_common_label = Counter(data[target]).most_common(1)[0][0]
            return most_common_label

        best_feature = features[0]
        tree = {best_feature: {}}

        for value in np.unique(data[best_feature]):
            subset = data[data[best_feature] == value]
            if subset.empty:
                most_common_label = Counter(data[target]).most_common(1)[0][0]
                tree[best_feature][value] = most_common_label
            else:
                tree[best_feature][value] = self.fit(subset, target, [f for f in features if f != best_feature], depth + 1)

        self.tree = tree
        return tree

    def predict_row(self, row, tree):
        if not isinstance(tree, dict):
            return tree

        feature = next(iter(tree))
        if row[feature] in tree[feature]:
            return self.predict_row(row, tree[feature][row[feature]])
        else:
            return Counter(self.tree).most_common(1)[0][0]

    def predict(self, data):
        return data.apply(lambda row: self.predict_row(row, self.tree), axis=1)


class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, sample_size=0.8):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.trees = []

    def bootstrap_sample(self, data):
        indices = np.random.choice(data.index, size=int(len(data) * self.sample_size), replace=True)
        return data.loc[indices]

    def fit(self, data, target, features):
        for _ in range(self.n_trees):
            sample = self.bootstrap_sample(data)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(sample, target, features)
            self.trees.append(tree)

    def predict(self, data):
        tree_preds = np.array([tree.predict(data) for tree in self.trees])
        majority_votes = [Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(data.shape[0])]
        return majority_votes

# Function for preprocessing data
def preprocess_data(df):
    mappings = {
        'Gender': {'Female': 0, 'Male': 1},
        'City': {'Chicago': 0, 'Houston': 1, 'Los Angeles': 2, 'Miami': 3, 'New York': 4, 'San Francisco': 5},
        'Membership Type': {'Bronze': 0, 'Gold': 1, 'Silver': 2},
        'Satisfaction Level': {'Neutral': 0, 'Satisfied': 1, 'Unsatisfied': 2}
    }
    
    # Map columns based on the defined mappings
    for column, mapping in mappings.items():
        if column in df.columns:
            df[column] = df[column].map(mapping)

    # Convert 'Discount Applied' from boolean to 0/1 (if it's a boolean type)
    if 'Discount Applied' in df.columns and df['Discount Applied'].dtype == 'bool':
        df['Discount Applied'] = df['Discount Applied'].astype(int)

    return df

# Load model
model_file = r'random_forest_model_90.pkl'
try:
    rf_model = joblib.load(model_file)
except Exception as e:
    st.error(f"Error saat memuat model: {e}")
    st.stop()

# Sidebar untuk mengunggah file
st.sidebar.title("Segment Pro")
uploaded_file = st.sidebar.file_uploader("Unggah File CSV", type="csv")

# Define consistent soft color mapping for each segment
color_mapping = {
    'Neutral': '#a8d5ba',      # Soft Green
    'Satisfied': '#ffe066',    # Soft Yellow
    'Unsatisfied': '#c9a9e3'   # Soft Lavender
}

# Extract colors as a list for charts that require a palette
color_palette = [color_mapping[key] for key in ['Neutral', 'Satisfied', 'Unsatisfied']]

def generate_insight(data, feature, target='Prediction'):
    """
    Fungsi untuk menghasilkan insight otomatis berdasarkan distribusi data.
    """
    insights = []
    if feature == 'Gender':
        gender_dist = data.groupby([feature, target]).size().unstack(fill_value=0)
        female_dominant = gender_dist.loc['Female'].idxmax()
        male_dominant = gender_dist.loc['Male'].idxmax()
        
        insights.append(f"- Wanita (Female) lebih banyak berada dalam kategori **{female_dominant}**.")
        insights.append(f"- Pria (Male) cenderung lebih sering berada dalam kategori **{male_dominant}**.")
    
    elif feature == 'Membership Type':
        membership_dist = data.groupby([feature, target]).size().unstack(fill_value=0)
        dominant_types = membership_dist.idxmax(axis=1)
        
        for membership, dominant in dominant_types.items():
            insights.append(f"- Pelanggan dengan **{membership} Membership** lebih dominan dalam kategori **{dominant}**.")
    
    elif feature == 'Total Spend':
        avg_spend = data.groupby(target)['Total Spend'].mean().sort_values(ascending=False)
        for category, spend in avg_spend.items():
            insights.append(f"- Pelanggan di kategori **{category}** memiliki rata-rata pengeluaran ${spend:.2f}.")
    
    elif feature == 'Discount Applied':
    # Memeriksa apakah label True dan False tersedia dalam data
        discount_dist = data.groupby([feature, target]).size().unstack(fill_value=0)
    
        if True in discount_dist.index:
            discount_true_dominant = discount_dist.loc[True].idxmax()
            insights.append(f"- Pelanggan yang menggunakan diskon lebih sering berada dalam kategori **{discount_true_dominant}**.")
        else:
            insights.append("- Tidak ada data pelanggan yang mendapat diskon.")

        if False in discount_dist.index:
            discount_false_dominant = discount_dist.loc[False].idxmax()
            insights.append(f"- Pelanggan yang tidak menggunakan diskon lebih sering berada dalam kategori **{discount_false_dominant}**.")
        else:
            insights.append("- Tidak ada data pelanggan yang tidak mendapat diskon.")
    

    elif feature == 'City':
        city_dist = data.groupby([feature, target]).size().unstack(fill_value=0)
        for city in city_dist.index:
            dominant = city_dist.loc[city].idxmax()
            insights.append(f"- Pelanggan di kota **{city}** lebih banyak berada dalam kategori **{dominant}**.")
    
    return insights

def additional_visualizations(data):
    st.write("### Hubungan Antar Fitur dengan Hasil Klasifikasi dan Insight")
    features = ['Gender', 'Membership Type', 'Total Spend', 'Discount Applied', 'City']
    for feature in features:
        if feature in data.columns:
            st.write(f"#### Pengaruh {feature} terhadap Hasil Klasifikasi")
            
            # Visualisasi dengan countplot
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(
                data=data,
                x=feature,
                hue='Prediction',
                palette=color_palette,
                ax=ax
            )
            plt.xticks(rotation=85)
            st.pyplot(fig)
            
            # Menghasilkan insight otomatis
            st.write("##### Insight:")
            insights = generate_insight(data, feature)
            for insight in insights:
                st.write(insight)

# Tambahkan bagian ini pada main loop Anda setelah `additional_visualizations` dipanggil.
def segment_analysis(data):
    st.write("### Analisis Segmentasi")
    st.write("Berikut adalah beberapa insight berdasarkan data segmentasi yang tersedia:")
    
    # Distribusi segmen
    segment_counts = data['Prediction'].value_counts()
    for segment, count in segment_counts.items():
        st.write(f"- **{segment}**: {count} pelanggan")

    # Insight tambahan (jika ada logika tambahan)
    st.write("""
    - Pelanggan dengan kepuasan **Satisfied** cenderung lebih loyal dan memiliki potensi lebih besar untuk upselling.
    - Pelanggan dengan kepuasan **Unsatisfied** memerlukan perhatian khusus untuk meningkatkan pengalaman mereka.
    """)

# Fungsi untuk menampilkan rekomendasi
def show_recommendations(segment):
    if segment == 'Satisfied':
        st.write("""
        **Rekomendasi untuk Pelanggan dengan Kepuasan Netral:**
        - **Promosi Eksklusif**: Tawarkan program loyalitas dengan manfaat seperti pengumpulan poin lebih cepat, diskon khusus, atau hadiah untuk pembelian berikutnya.
        - **Upselling dan Cross-Selling**: Berikan rekomendasi produk pelengkap atau premium berdasarkan histori pembelian.
        - **Referral Program: Jadikan pelanggan sebagai duta merek dengan insentif untuk setiap rekomendasi atau ulasan positif.
        - **Promosi Lokasi: Perluas pasar di kota dengan dominasi segmen ini (misalnya, San Francisco dan New York) untuk menarik pelanggan baru.
        """)
        
    elif segment == 'Neutral':
        st.write("""
        **Rekomendasi untuk Pelanggan dengan Kepuasan Tinggi:**
        - **Promosi Eksklusif**: Perkenalkan insentif diskon atau hadiah yang relevan untuk meningkatkan keterlibatan pelanggan dalam pembelian berikutnya.
        - **Upselling dan Cross-Selling**: Dorong pembelian produk tambahan yang sesuai dengan histori pembelian pelanggan.
        - **Referral Program**: Ajak pelanggan netral untuk merekomendasikan teman mereka dengan memberikan diskon untuk transaksi berikutnya.
        - **Promosi Lokasi**: Fokus pada kota dengan banyak pelanggan netral (misalnya, Chicago dan Los Angeles) untuk meningkatkan pengalaman dan kepuasan.
        """)
        
    elif segment == 'Unsatisfied':
        st.write("""
        **Rekomendasi untuk Pelanggan Tidak Puas:**
        - **Promosi Eksklusif**: Hubungi pelanggan dengan penawaran eksklusif, seperti diskon besar atau voucher gratis untuk pembelian berikutnya.
        - **Upselling dan Cross-Selling**: Tawarkan produk dengan harga ekonomis yang sesuai dengan preferensi mereka, berdasarkan data pembelian.
        - **Referral Program**: Libatkan pelanggan dengan program referensi, sambil memberikan insentif tambahan jika mereka kembali bertransaksi.
        - **Promosi Lokasi**: Tingkatkan upaya pemasaran di kota dengan banyak pelanggan tidak puas (misalnya, Houston dan Miami) untuk menarik minat mereka kembali.
        """)

# Fungsi untuk mengembalikan nilai asli (inverse mapping)
def inverse_preprocess_data(df):
    inverse_mappings = {
        'Gender': {0: 'Female', 1: 'Male'},
        'City': {0: 'Chicago', 1: 'Houston', 2: 'Los Angeles', 3: 'Miami', 4: 'New York', 5: 'San Francisco'},
        'Membership Type': {0: 'Bronze', 1: 'Gold', 2: 'Silver'},
        'Discount Applied': {0: False, 1: True},
        'Satisfaction Level': {0: 'Neutral', 1: 'Satisfied', 2: 'Unsatisfied'}
    }

    for col, inverse_mapping in inverse_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(inverse_mapping)

    return df

# Main Interface
st.title("Segmentasi Pelanggan Berdasarkan Tingkat Kepuasan")
st.write("""
Aplikasi ini menampilkan segmen pelanggan berdasarkan prediksi kepuasan dan rekomendasi untuk setiap segmen.
Silakan pilih segmen untuk melihat detail ID pelanggan dan rekomendasi.
""")

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)

        if data.empty:
            st.error("Dataset yang diunggah kosong.")
        else:
            # Preprocessing
            data_preprocessed = preprocess_data(data)

            # Melakukan prediksi
            try:
                features = ['Gender', 'Membership Type', 'Total Spend', 'Discount Applied', 'City']
                predictions = rf_model.predict(data_preprocessed[features])

                # Konversi hasil prediksi ke label yang sesuai
                label_mapping = {0: 'Neutral', 1: 'Satisfied', 2: 'Unsatisfied'}
                predictions_labels = [label_mapping.get(pred, 'Neutral') for pred in predictions]

                # Menambahkan kolom Customer ID dan Prediksi
                data_preprocessed['Prediction'] = predictions_labels
                data_preprocessed['Customer ID'] = data['Customer ID']

                # Mengembalikan nilai asli (kata-kata) untuk analisis
                data_analyzed = inverse_preprocess_data(data_preprocessed.copy())

                # Pilihan untuk mengunduh hasil
                csv = data_preprocessed[['Customer ID', 'Prediction']].to_csv(index=False)
                st.download_button(
                    label="Unduh Hasil Prediksi",
                    data=csv,
                    file_name="hasil_klasifikasi_segmentasi.csv",
                    mime="text/csv",
                )

                # Analisis Chart: Distribusi Segmentasi
                st.write("### Analisis Distribusi Segmentasi:")
                segment_counts = data_preprocessed['Prediction'].value_counts()

                # Plot Pie Chart
                fig, ax = plt.subplots()
                ax.pie(
                    segment_counts, 
                    labels=segment_counts.index, 
                    autopct='%1.1f%%', 
                    startangle=90, 
                    colors=[color_mapping[label] for label in segment_counts.index]
                )
                ax.axis('equal')
                st.pyplot(fig)

                # Membuat Bar Chart dengan warna dan menambahkan tulisan "count"
                fig, ax = plt.subplots()
                bars = ax.bar(segment_counts.index, segment_counts.values, color=[color_mapping[label] for label in segment_counts.index])
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom')

                ax.set_xlabel('Segmen Kepuasan')
                ax.set_ylabel('Jumlah Pelanggan')
                ax.set_title('Distribusi Pelanggan Berdasarkan Kepuasan')
                st.pyplot(fig)

                # Tampilkan ID pelanggan untuk semua segmen dalam tabel terpisah
                st.write("### ID Pelanggan Berdasarkan Segmen")
                segments = ['Neutral', 'Satisfied', 'Unsatisfied']
                columns = st.columns(len(segments))

                for col, segment in zip(columns, segments):
                    selected_ids = data_preprocessed[data_preprocessed['Prediction'] == segment]['Customer ID']
                    with col:
                        st.write(f"**{segment}**")
                        segment_table = pd.DataFrame({'ID': selected_ids.values})
                        st.dataframe(segment_table, height=200)

                # Mengembalikan nilai asli untuk analisis lebih lanjut
                data_analyzed = inverse_preprocess_data(data_preprocessed)

                # Visualisasi tambahan
                additional_visualizations(data_analyzed)

                # Gabungkan Analisis Kebutuhan Per Kelas dan Rekomendasi
                st.write("### Analisis dan Rekomendasi Berdasarkan Segmen")
                combined_choice = st.selectbox("Pilih Segmen untuk Analisis dan Rekomendasi", ['Neutral', 'Satisfied', 'Unsatisfied'])

                if combined_choice:
                    segment_data = data_analyzed[data_analyzed['Prediction'] == combined_choice]

                    # Analisis Kebutuhan Per Kelas
                    st.write(f"#### Analisis Kebutuhan untuk Segmen: {combined_choice}")
                    if not segment_data.empty:
                        avg_spending = segment_data['Total Spend'].mean()
                        avg_items = segment_data['Items Purchased'].mean()
                        avg_rating = segment_data['Average Rating'].mean()

                        st.write(f"- **Rata-rata pengeluaran**: ${avg_spending:.2f}")
                        st.write(f"- **Rata-rata jumlah barang dibeli**: {avg_items:.1f}")
                        st.write(f"- **Rata-rata rating**: {avg_rating:.1f}")
                    else:
                        st.warning(f"Tidak ada data untuk segmen {combined_choice}.")

                    # Rekomendasi Pelanggan
                    st.write(f"#### Rekomendasi untuk Segmen: {combined_choice}")
                    show_recommendations(combined_choice)

            except Exception as e:
                st.error(f"Error saat melakukan prediksi: {e}")
    except Exception as e:
        st.error(f"Error saat membaca file CSV: {e}")
