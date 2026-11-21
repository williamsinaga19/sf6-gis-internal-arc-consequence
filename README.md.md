# Pemodelan Konsekuensi Pelepasan Gas Beracun ($\text{HF}$ & $\text{SO}_2$) dan Risiko Asfiksia Akibat Gangguan *Internal Arc* pada GIS $\text{SF}_6$ 150 kV

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/williamsinaga19/sf6-gis-internal-arc-consequence/blob/main/SF6_GIS_Internal_Arc_Consequence_Model.ipynb)

* **Penulis:** William Ishak Sinaga (NPM: 2506564323)
* **Program Studi:** Magister Teknik Kimia — Konsentrasi Manajemen Keselamatan Proses, Universitas Indonesia
* **Mata Kuliah:** Analisis Data Keselamatan Proses (Tugas Akhir / UAS)

---

Repositori ini berisi model kuantitatif berbasis *first-principles* (termodinamika, kinetika reaksi kimia, neraca massa dinamis, dan toksikologi *Probit*) yang dikombinasikan dengan **simulasi Monte Carlo** untuk menganalisis konsekuensi keselamatan proses akibat gangguan busur api internal (*internal arc*) dan pecahnya *bursting disc* atau *burn-through* pada *Gas-Insulated Switchgear* (GIS) tegangan tinggi 150 kV di dalam ruangan.

Seluruh simulasi dikembangkan menggunakan Python di **Google Colaboratory** hanya dengan pustaka komputasi ilmiah standar (`numpy`, `scipy`, `pandas`, `matplotlib`) sehingga dapat direproduksi langsung tanpa instalasi paket tambahan.

## 1. Latar Belakang & Tujuan

Gas sulfur heksafluorida ($\text{SF}_6$) secara murni bersifat inert dan tidak beracun, namun memiliki densitas sekitar 5 kali lebih berat daripada udara sehingga berpotensi menyebabkan asfiksia apabila terakumulasi di ruang tertutup. Saat terjadi gangguan *internal arc* berenergi tinggi pada kompartemen GIS, $\text{SF}_6$ terdekomposisi secara termal membentuk prekursor reaktif seperti $\text{SF}_4$ dan $\text{SOF}_2$. Begitu *bursting disc* pecah atau terjadi *burn-through* pada *enclosure*, gas terurai ini menyembur keluar dan bereaksi cepat dengan uap air udara ruangan (hidrolisis) menghasilkan gas korosif dan sangat beracun **Hidrogen Fluorida ($\text{HF}$)** serta **Sulfur Dioksida ($\text{SO}_2$)**.

Model ini bertujuan untuk:

1. Membandingkan tingkat bahaya akut antara **toksisitas gas hasil dekomposisi ($\text{HF}$, $\text{SO}_2$)** terhadap **risiko asfiksia ($\text{O}_2 \le 16\%$)** pada gedung GIS berukuran besar.
2. Mengevaluasi dampak kegagalan proteksi (*clearing time* normal $50\text{ ms}$ vs. *burn-through* $300\text{ ms}$), ketidakpastian kinetika hidrolisis $\text{SOF}_2$, serta keandalan sistem *emergency exhaust*.
3. Mengkuantifikasi ketidakpastian risiko secara probabilistik (Monte Carlo, $N = 1.000$) dan mengidentifikasi parameter paling sensitif melalui analisis korelasi peringkat Spearman (*Tornado Chart*).

## 2. Arsitektur & Formulasi Matematis

Model dibagi menjadi 4 modul terkopel yang diselesaikan secara numerik:

### Modul 1: *Source Term* (*Internal Arc* & *Real-Gas Orifice Blowdown*)

Energi total busur listrik ($E_{\text{arc}}$) dan total mol produk dekomposisi primer ($N_{\text{dec},0}$) menggunakan *yield* empiris elektroda aluminium ($G_{\text{Al}}$):

$$
E_{\text{arc}} = U_{\text{arc}} \cdot I_{\text{fault}} \cdot t_{\text{arc}}, \qquad N_{\text{dec},0} = G_{\text{Al}} \cdot E_{\text{arc}}
$$

Volume internal kompartemen dihitung dari massa gas $\text{SF}_6$ dan tekanan pengisian nominal dengan koreksi faktor kompresibilitas gas nyata ($Z_{\text{SF}_6} = 0{,}935$ pada $600\text{ kPa}$, $30^\circ\text{C}$):

$$
V_{\text{GIS}} = \frac{Z_{\text{SF}_6} \, N_{\text{SF}_6,0} \, R \, T_1}{P_1}
$$

Laju pelepasan massa gas dari kompartemen ke ruangan ($\dot{m}_{\text{out}}$) mengikuti persamaan aliran *orifice* CCPS CPQRA (subsonik dan *choked/sonic flow*) hingga tekanan kompartemen $P_1(t)$ turun menyamai tekanan atmosfer $P_2 = P_{\text{atm}}$:

$$
\frac{P_{\text{choked}}}{P_1} = \left(\frac{2}{\gamma + 1}\right)^{\frac{\gamma}{\gamma - 1}}
$$

* **Aliran *Choked* (Sonik):**

$$
\dot{m}_{\text{out}}(t) = C_D A_{\text{orifice}} P_1(t) \sqrt{\frac{\gamma M_{\text{SF}_6}}{Z_{\text{SF}_6} R T_1} \left(\frac{2}{\gamma + 1}\right)^{\frac{\gamma + 1}{\gamma - 1}}}
$$

* **Aliran *Non-Choked* (Subsonik):**

$$
\dot{m}_{\text{out}}(t) = C_D A_{\text{orifice}} P_1(t) \sqrt{\frac{2 M_{\text{SF}_6}}{Z_{\text{SF}_6} R T_1} \frac{\gamma}{\gamma - 1} \left[ \left(\frac{P_2}{P_1(t)}\right)^{\frac{2}{\gamma}} - \left(\frac{P_2}{P_1(t)}\right)^{\frac{\gamma+1}{\gamma}} \right]}
$$

### Modul 2: Neraca Massa Ruangan Terstratifikasi & Kinetika Hidrolisis (Sistem 6 ODE)

Karena densitas $\text{SF}_6$ jauh lebih besar daripada udara, pencampuran dimodelkan pada **volume efektif zona bawah pernapasan** $V_{\text{eff}} = f_{\text{strat}} \cdot V_{\text{room}}$. Dua reaksi hidrolisis orde-2 berurutan di udara lembap ruangan:

1. $\text{SF}_4 + \text{H}_2\text{O} \xrightarrow{k_1^*} \text{SOF}_2 + 2\text{HF}$
2. $\text{SOF}_2 + \text{H}_2\text{O} \xrightarrow{k_2^*} \text{SO}_2 + 2\text{HF}$

Konstanta laju molekuler ($k_i$, $\text{cm}^3/\text{s}$) dikonversi ke satuan molar SI ($\text{m}^3/(\text{mol}\cdot\text{s})$) menggunakan bilangan Avogadro: $k_i^* = k_i \cdot N_A \cdot 10^{-6}$. Sistem 6 Persamaan Diferensial Biasa (ODE) diselesaikan dengan *stiff solver* **`Radau`** (`scipy.integrate.solve_ivp`) mengingat rentang skala waktu reaksi yang sangat kontras (detik vs. jam):

$$
\frac{dN_{\text{tank}}}{dt} = -\dot{n}_{\text{out}}(t)
$$

$$
\frac{dC_{\text{SF}_6}}{dt} = \frac{y_{\text{SF}_6} \dot{n}_{\text{out}}(t)}{V_{\text{eff}}} - \lambda_{\text{vent}}(t) C_{\text{SF}_6}
$$

$$
\frac{dC_{\text{SF}_4}}{dt} = \frac{y_{\text{SF}_4} \dot{n}_{\text{out}}(t)}{V_{\text{eff}}} - k_1^* C_{\text{SF}_4} C_{\text{H}_2\text{O}} - \lambda_{\text{vent}}(t) C_{\text{SF}_4}
$$

$$
\frac{dC_{\text{SOF}_2}}{dt} = \frac{y_{\text{SOF}_2} \dot{n}_{\text{out}}(t)}{V_{\text{eff}}} + k_1^* C_{\text{SF}_4} C_{\text{H}_2\text{O}} - k_2^* C_{\text{SOF}_2} C_{\text{H}_2\text{O}} - \lambda_{\text{vent}}(t) C_{\text{SOF}_2}
$$

$$
\frac{dC_{\text{SO}_2}}{dt} = k_2^* C_{\text{SOF}_2} C_{\text{H}_2\text{O}} - \lambda_{\text{vent}}(t) C_{\text{SO}_2}
$$

$$
\frac{dC_{\text{HF}}}{dt} = 2 k_1^* C_{\text{SF}_4} C_{\text{H}_2\text{O}} + 2 k_2^* C_{\text{SOF}_2} C_{\text{H}_2\text{O}} - \left(\lambda_{\text{vent}}(t) + k_{\text{wall}}\right) C_{\text{HF}}
$$

dengan laju ventilasi dinamis $\lambda_{\text{vent}}(t) = \text{ACH}(t)/3600\text{ s}^{-1}$ yang beralih dari ventilasi normal ke *emergency exhaust* pada $t = 120\text{ s}$.

### Modul 3: Evaluasi Toksikologi (*Toxic Load*, AEGL, & Multi-Model *Probit*)

Konsentrasi dievaluasi terhadap ambang darurat akut **NRC AEGL-1/2/3**, **NIOSH IDLH**, dan batas asfiksia ($\text{O}_2 \le 16\%$). Akumulasi *Toxic Load* ($TL$) dihitung menggunakan integrasi numerik trapesium ($t$ dalam menit) dan dikonversi menjadi probabilitas fatalitas ($P_f$) melalui fungsi distribusi normal kumulatif standar $\Phi$:

$$
TL_j(t) = \int_{0}^{t} \left[C_j(\tau)\right]^{n_j} \, d\tau, \qquad Y_j(t) = A_j + B_j \ln\left(TL_j(t)\right), \qquad P_{f,j}(t) = \Phi\left(Y_j(t) - 5\right)
$$

Probabilitas fatalitas gabungan dihitung dengan asumsi independensi efek toksik: $P_{f,\text{total}} = 1 - (1 - P_{f,\text{HF}})(1 - P_{f,\text{SO}_2})$.

Untuk menghindari bias akibat perbedaan referensi toksikologi (termasuk *typo* historis pada konstanta $\text{HF}$ edisi lama CCPS 1989), model menyajikan **3 pembanding *Probit* untuk $\text{HF}$**:

* **Model Utama — TNO Green Book / BEVI (2009):** $A = -8{,}620$, $B = 1{,}000$, $n = 1{,}50$ ($C$ dalam $\text{mg/m}^3$, standar QRA Eropa).
* **Pembanding 1 — CCPS Terkoreksi (Casal, 2008):** $A = -25{,}870$, $B = 3{,}354$, $n = 1{,}00$ ($C$ dalam $\text{mg/m}^3$).
* **Pembanding 2 (Batas Atas Konservatif) — Turunan NRC AEGL (2004):** $n = 2{,}00$ diturunkan dari 1-h $\text{LC}_{50}$ spesies tersensitif ($342\text{ ppm}$) dan 1-h AEGL-3 ($44\text{ ppm}$) menggunakan metode James (2014).
* **Model $\text{SO}_2$ — CCPS:** $A = -15{,}670$, $B = 2{,}100$, $n = 1{,}00$ ($C$ dalam $\text{ppm}$).

### Modul 4: Simulasi Monte Carlo & Analisis Sensitivitas

Sebanyak $N = 1.000$ iterasi dijalankan dengan *fixed seed* (`42`) untuk menjamin *reproducibility*. Pengaruh masing-masing parameter input terhadap probabilitas fatalitas saat evakuasi dievaluasi menggunakan **korelasi peringkat Spearman ($\rho$)** dan divisualisasikan dalam bentuk *Tornado Chart*.

## 3. Spesifikasi Skenario & Distribusi Input

Objek studi yang dianonimkan merepresentasikan **gedung GIS 150 kV (5 Bay)** dengan konfigurasi *enclosure* 3-fasa berbahan paduan aluminium/baja dan konduktor utama aluminium.

### Parameter *Baseline* Deterministik

| Parameter | Nilai *Baseline* | Keterangan / Rujukan |
| :--- | :--- | :--- |
| Volume ruang GIS ($V_{\text{room}}$) | $2.860{,}32\text{ m}^3$ | Faktor stratifikasi *baseline* $f_{\text{strat}} = 0{,}25$ ($V_{\text{eff}} = 715{,}08\text{ m}^3$) |
| Massa $\text{SF}_6$ kompartemen CB 3-fasa | $59\text{ kg}$ | Tekanan pengisian nominal $P_1 = 600\text{ kPa abs}$ |
| Arus hubung singkat ($I_{\text{fault}}$) & tegangan busur ($U_{\text{arc}}$) | $40\text{ kA rms}$, $500\text{ V}$ | Rating ketahanan hubung singkat tipikal GIS 150 kV |
| Durasi busur ($t_{\text{arc}}$) | $50\text{ ms}$ vs. $300\text{ ms}$ | Pemutusan normal ($\le 50\text{ ms}$) vs. ambang *burn-through* ($>300\text{ ms}$) |
| Diameter *orifice* ($D$) & $C_D$ | $50\text{ mm}$, $C_D = 0{,}61$ | *Bursting disc* / *sharp-edged orifice* (CCPS, 2000) |
| Kondisi udara ruangan | $30^\circ\text{C}$, $\text{RH } 75\%$ | Iklim tropis ($[\text{H}_2\text{O}] \approx 1{,}26\text{ mol/m}^3$) |
| Laju ventilasi ($\text{ACH}$) | $1{,}0 \rightarrow 6{,}0\text{ ACH}$ | Aktivasi *emergency exhaust* pada $t = 2\text{ menit}$ (gagal: $0{,}2\text{ ACH}$) |

### Distribusi 8 Variabel Stokastik (Monte Carlo, $N = 1.000$)

| Variabel Input | Distribusi | Parameter / Rentang |
| :--- | :--- | :--- |
| Arus gangguan ($I_{\text{fault}}$) | *Triangular* | Min $15$, Mode $30$, Max $40\text{ kA}$ |
| Durasi busur ($t_{\text{arc}}$) | *Triangular* | Min $50$, Mode $300$, Max $500\text{ ms}$ |
| *Yield* dekomposisi Al ($G_{\text{Al}}$) | *Triangular* | Min $100$, Mode $600$, Max $650\text{ nmol/J}$ |
| Fraksi awal $\text{SF}_4$ ($\alpha$) | *Uniform* | $0{,}20\text{ – }0{,}80$ |
| Konstanta hidrolisis $\text{SOF}_2$ ($k_2$) | *Log-Uniform* | $1{,}2 \times 10^{-23}\text{ – }4{,}0 \times 10^{-20}\text{ cm}^3/\text{s}$ |
| Faktor stratifikasi ruang ($f_{\text{strat}}$) | *Uniform* | $0{,}20\text{ – }0{,}50$ |
| Ventilasi darurat ($\text{ACH}_{\text{emerg}}$) | *Mixture* (Bernoulli + Uniform) | $15\%$ gagal ($0{,}2\text{ ACH}$); $85\%$ beroperasi ($4\text{ – }8\text{ ACH}$) |
| Waktu evakuasi operator ($t_{\text{evac}}$) | *Triangular* | Min $2$, Mode $10$, Max $30\text{ menit}$ |

## 4. Ringkasan Hasil Simulasi

### A. Perbandingan 4 Skenario Deterministik

| Skenario | $E_{\text{arc}}$ (MJ) | Peak $\text{HF}$ (ppm) | Peak $\text{SO}_2$ (ppm) | Min $\text{O}_2$ (%) | $P_f$ 10m BEVI (%) | $P_f$ 30m BEVI (%) | $P_f$ 30m CCPS (%) | $P_f$ 30m AEGL-$n2$ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1:** Proteksi Normal ($50\text{ ms}$) | 1,0 | 17,1 | 0,0 | 20,71 | 0,00 | 0,00 | 0,00 | 0,00 |
| **S2:** *Burn-Through* ($300\text{ ms}$, Gas-Phase $k_2$) | 6,0 | 102,6 | 0,2 | 20,71 | 0,00 | 0,00 | 0,00 | 0,43 |
| **S3:** *Burn-Through* ($300\text{ ms}$, Surface $k_2$) | 6,0 | 294,3 | 96,7 | 20,71 | 0,02 | 0,05 | 0,00 | 7,61 |
| **S4:** *Burn-Through* ($300\text{ ms}$, Exhaust Gagal) | 6,0 | 308,3 | 102,7 | 20,71 | 0,10 | 2,28 | 13,73 | 28,48 |

### B. Ringkasan Statistik Monte Carlo ($N = 1.000$)

| Metrik Output | Mean | P5 (Optimis) | P50 (Median) | P95 (Konservatif) | Max |
| :--- | :---: | :---: | :---: | :---: | :---: |
| $E_{\text{arc}}$ (MJ) | 4,035 | 1,676 | 3,973 | 6,608 | 8,208 |
| Peak $\text{HF}$ (ppm) | 59,889 | 12,961 | 47,691 | 149,716 | 504,966 |
| Peak $\text{SO}_2$ (ppm) | 12,054 | 0,136 | 5,092 | 42,242 | 140,878 |
| Min $\text{O}_2$ (%) | 20,764 | 20,667 | 20,777 | 20,826 | 20,829 |
| $P_f$ Evakuasi — TNO BEVI 2009 (%) | 0,003 | 0,000 | 0,000 | 0,001 | 1,168 |
| $P_f$ Evakuasi — NRC AEGL $n=2$ (%) | 0,366 | 0,000 | 0,029 | 1,538 | 26,661 |

**Probabilitas Pelampauan Ambang Bahaya:**

* Peak $\text{HF} \ge$ NIOSH IDLH ($30\text{ ppm}$): **72,5%**
* Peak $\text{HF} \ge$ AEGL-2 10 menit ($95\text{ ppm}$): **16,2%**
* Peak $\text{HF} \ge$ AEGL-3 10 menit ($170\text{ ppm}$): **2,9%**
* Fatalitas $\ge 1\%$ saat evakuasi: **0,1%** (TNO BEVI 2009) / **7,6%** (batas atas NRC AEGL $n=2$)
* Defisiensi $\text{O}_2 \le 16\%$ (Asfiksia): **0,0%**

### C. Temuan Utama Rekayasa Keselamatan Proses

1. **Toksisitas $\text{HF}$ Jauh Lebih Kritis daripada Asfiksia:** Pada ruang GIS besar ($2.860{,}32\text{ m}^3$), pelepasan satu kompartemen CB tidak pernah memicu defisiensi oksigen (min. $\text{O}_2 = 20{,}67\%$). Sebaliknya, **72,5%** skenario melampaui batas NIOSH IDLH $\text{HF}$ ($30\text{ ppm}$) dan **16,2%** menembus AEGL-2 ($95\text{ ppm}$, ambang gangguan kemampuan evakuasi).
2. **Keandalan *Emergency Exhaust* Sangat Menentukan:** Pada skenario *burn-through* (S3 vs. S4), berfungsinya sistem ventilasi darurat ($6\text{ ACH}$) memangkas probabilitas fatalitas 30 menit (TNO BEVI 2009) lebih dari **97%** (dari $2{,}28\%$ turun menjadi $0{,}05\%$).
3. **Hierarki Pengendalian Berdasarkan *Tornado Chart*:** Ketidakpastian epistemik terbesar berasal dari laju hidrolisis $\text{SOF}_2$ ($\log k_2$, $\rho = +0{,}48$). Dari sisi parameter yang dapat dikendalikan secara rekayasa, **durasi pemutusan busur ($t_{\text{arc}}$, $\rho = +0{,}40$)** merupakan faktor paling dominan—hampir **2 kali lebih berpengaruh** dibandingkan kapasitas ventilasi darurat ($\text{ACH}$, $\rho = -0{,}21$) dan waktu evakuasi operator ($t_{\text{evac}}$, $\rho = +0{,}20$). Menjaga keandalan sistem proteksi untuk mencegah *burn-through* adalah mitigasi paling efektif.

## 5. Batasan Model (*Limitations*)

Untuk transparansi ilmiah, pengguna perlu memperhatikan empat batasan utama dalam model ini:

1. ***Isothermal Nominal-Pressure Blowdown*:** Pelepasan gas dihitung dari tekanan pengisian dingin konstan ($600\text{ kPa}$, $30^\circ\text{C}$) tanpa memodelkan lonjakan tekanan dan temperatur sesaat akibat pemanasan plasma busur api di dalam kompartemen.
2. **Model Kotak Terstratifikasi 0D (*Lumped Box Model*):** Ruangan dimodelkan tercampur merata di dalam zona bawah $V_{\text{eff}}$. Model ini cocok untuk *screening* CPQRA, namun tidak menangkap gradien spasial area dekat kebocoran (*near-field*) atau akumulasi lokal di parit kabel (*cable trench*) sebagaimana simulasi CFD 3D.
3. **Toksisitas Inhalasi Langsung $\text{SOF}_2$ & Debu $\text{AlF}_3$:** Evaluasi *Probit* difokuskan pada $\text{HF}$ dan $\text{SO}_2$. Efek inhalasi langsung $\text{SOF}_2$ yang terhidrolisis di saluran pernapasan serta debu padat $\text{AlF}_3$ (yang dapat mengadsorpsi gas asam) belum dimasukkan ke dalam fungsi *Probit*.
4. **Asumsi Konservatif *Wall Loss* ($k_{\text{wall}} = 0$):** Reaksi adsorpsi/korosi $\text{HF}$ pada permukaan logam dan dinding ruangan diabaikan sehingga hasil konsentrasi $\text{HF}$ merepresentasikan batas atas konservatif.

## 6. Struktur Repositori & Cara Menjalankan di Google Colab

```text
sf6-gis-internal-arc-consequence/
├── README.md                                       # Dokumentasi model, persamaan, & ringkasan hasil
└── SF6_GIS_Internal_Arc_Consequence_Model.ipynb    # Notebook Google Colab (output & grafik tersimpan)
```

1. Klik tombol **Open in Colab** di bagian atas halaman ini (atau buka file `SF6_GIS_Internal_Arc_Consequence_Model.ipynb` di Google Colab).
2. Klik menu **Runtime** $\rightarrow$ **Restart session and run all**.
3. Tidak diperlukan instalasi pustaka tambahan (`!pip install`). Seluruh 9 sel (simulasi deterministik 4 skenario + Monte Carlo $N = 1.000$ + 6 panel visualisasi) akan selesai dieksekusi dalam waktu sekitar **20–30 detik**.
4. Untuk menguji konfigurasi GIS lain, cukup ubah parameter di dalam *dataclass* `GISConfig` pada **Cell 2**.

## 7. Daftar Pustaka

* **CCPS (2000).** *Guidelines for Chemical Process Quantitative Risk Analysis (CPQRA)*, 2nd Edition. Center for Chemical Process Safety, AIChE.
* **Chu, F. Y. (1986).** $\text{SF}_6$ Decomposition in Gas-Insulated Equipment. *IEEE Transactions on Electrical Insulation*, EI-21(5), 693–725.
* **IEEE Std C37.122.3-2024.** *IEEE Guide for Sulfur Hexafluoride ($\text{SF}_6$) Gas Handling for High-Voltage (Over 1000 Vac) Equipment*.
* **James, M. (2014).** Simplified Methods of Using Probit Analysis in Consequence Analysis. *Process Safety Progress*, AIChE.
* **NRC (2004).** *Acute Exposure Guideline Levels for Selected Airborne Chemicals: Volume 4 — Hydrogen Fluoride*. National Research Council, National Academies Press.
* **NRC (2010).** *Acute Exposure Guideline Levels for Selected Airborne Chemicals: Volume 8 — Sulfur Dioxide*. National Research Council, National Academies Press.
* **RIVM / BEVI (2009).** *Reference Manual Bevi Risk Assessments (TNO Green Book Probit Relations)*. National Institute for Public Health and the Environment, Netherlands.
* **Van Brunt, R. J., & Sauers, I. (1986).** Gas-phase hydrolysis of $\text{SOF}_2$ and $\text{SOF}_4$. *The Journal of Chemical Physics*, 85(8), 4377–4380.