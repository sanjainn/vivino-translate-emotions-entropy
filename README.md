1️⃣ Open Anaconda Prompt

Then go to your project folder:

cd path\to\vivino-translate-emotions-entropy


(Example: cd C:\Users\Sanil\projects\vivino-translate-emotions-entropy)

2️⃣ Create a fresh conda environment
conda create -n vivino-env python=3.10 -y

3️⃣ Activate the environment
conda activate vivino-env


You should now see:

(vivino-env)

4️⃣ Upgrade pip (important)
python -m pip install --upgrade pip

5️⃣ Install dependencies from requirements.txt
pip install -r requirements.txt


⚠️ This step will take time (Torch + Transformers).

6️⃣ Run the app locally (test)
streamlit run app.py


Browser should open → app should load.

✅ If this works, your repo is portable.

Quick Clone test late strategy:

git clone https://github.com/sanjainn/vivino-translate-emotions-entropy.git
cd vivino-translate-emotions-entropy
conda create -n vivino-env python=3.10 -y
conda activate vivino-env
pip install -r requirements.txt
streamlit run vivino_compare_vintages_selenium.py
