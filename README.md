# Final-experimentation-program-1620

Predicts concrete mix proportions (**water content, FA, CA, 10 mm, 20 mm**) from aggregate shape parameters
(**A** = Angularity, **S** = Sphericity, **T** = Texture), target **Slump**, **w/c ratio** and, optionally, **Compaction Factor (CF)**.

Models: one `GradientBoostingRegressor` per output, trained twice (with CF and without CF).
All results are printed in the terminal and all graphs are saved to disk.

---

## Project structure

```
Final-experimentation-program-1620/
├── Concrete_Input-COMBINED-1620.xlsx   # input dataset (read and appended to)
├── concrete_mix_regression.py          # main program
├── requirements.txt                    # Python dependencies
├── README.md
├── .gitignore
└── outputs/                            # created automatically on first run
    ├── graphs/
    │   ├── shape_effects/              # A, S, T vs Slump / CF
    │   ├── r2_scores/                  # R² bar charts (with / without CF)
    │   └── actual_vs_predicted/        # one image per output variable
    ├── models/                         # trained models (.pkl)
    ├── metrics/                        # R², MAE, MSE as CSV
    └── backups/                        # Excel backup before every save
```

## Excel file requirements

`Concrete_Input-COMBINED-1620.xlsx` must be in the same folder as the script and contain these columns:

| Type    | Columns                                                  |
|---------|----------------------------------------------------------|
| Inputs  | `A`, `S`, `T`, `Slump`, `CF`, `w/c_ratio`                |
| Outputs | `water_content`, `FA`, `CA`, `10`, `20`                  |
| Other   | `Sr. No.` (auto-incremented when a new row is added)     |

## Setup

Requires **Python 3.9+**.

**Windows (PowerShell / VS Code terminal)**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> If PowerShell blocks activation, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## Run

```bash
python concrete_mix_regression.py
```

On start the program will:
1. Load the Excel data.
2. Train all models and print **R², MAE, MSE** for every output.
3. Save all graphs, models and metrics under `outputs/`.
4. Open a terminal menu:

```
1) Predict a new mix design (and save it to Excel)
2) Re-train models and regenerate all graphs
3) Show accuracy (R², MAE, MSE) again
4) Exit
```

### Option 1 – predicting a mix
1. Answer whether you have CF (y/n).
2. Enter A, S, T, target slump, w/c ratio (and CF if available).
3. The predicted mix is printed.
4. Optionally confirm/edit values (press **ENTER** to keep a value), then enter the **actual** slump and CF.
5. The row is appended to the Excel file. A timestamped backup of the previous file is stored in `outputs/backups/`.
6. You can retrain immediately so the new row is used by the models.

> Close the Excel file before saving – Windows will not allow writing to a file that is open.

## Configuration

Edit the constants at the top of `concrete_mix_regression.py`:

| Setting        | Default | Meaning                                         |
|----------------|---------|-------------------------------------------------|
| `TEST_SIZE`    | `0.2`   | Fraction of data used for testing               |
| `RANDOM_STATE` | `42`    | Seed for reproducible splits                    |
| `N_ESTIMATORS` | `100`   | Trees per Gradient Boosting model               |
| `POLY_DEGREE`  | `2`     | Degree of the red trend curve in scatter plots  |
| `DPI`          | `200`   | Resolution of saved graphs                      |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Excel file not found` | Keep the `.xlsx` in the same folder as the script, with the exact name. |
| `Excel is missing columns` | Check column names match the table above. |
| `PermissionError` on save | Close the Excel file and retry. |
| `ModuleNotFoundError` | Activate `.venv` and run `pip install -r requirements.txt`. |

## Push to GitHub

```bash
# one-time setup inside the project folder
git init
git add .
git commit -m "Initial commit: local concrete mix regression program"
git branch -M main

# create an empty repo on GitHub first (no README), then:
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Later updates:
```bash
git add .
git commit -m "Describe your change"
git push
```

> **Data privacy:** the Excel file is research data. If it should not be public, create a **private** repo,
> or add `Concrete_Input-COMBINED-1620.xlsx` to `.gitignore` before the first commit.
