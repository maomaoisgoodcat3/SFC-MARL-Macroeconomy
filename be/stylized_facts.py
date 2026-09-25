"""
Kiem chung "stylized facts" (quy luat thuc nghiem lap lai trong du lieu kinh te
that) tren chuoi thoi gian training/simulate that cua du an -- dung khung da xac
minh tu literature trong phien lam viec truoc (Zheng et al. 2020/2022; Stock &
Watson 1999, dan lai qua ABIDES-Economist; Phillips 1958; Okun 1963).

Doc truc tiep tu file result.json cua Ray Tune (KHONG can checkpoint -- day la
phan tich hoi cuu tren time series da co san, khac voi be/benchmark.py can nap
lai policy de chay counterfactual).

Dung: python -m be.stylized_facts --result-json <duong_dan_result.json>
"""
import argparse
import json
import numpy as np


def load_result_json(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def extract_series(rows):
    """Trich cac chuoi vi mo tu custom_metrics (env_runners.custom_metrics.*_mean)."""
    series = {
        "unemployment": [], "inflation": [], "real_gdp": [], "gini": [],
        "avg_wage": [], "npl": [], "active_employees": [],
    }
    for r in rows:
        cm = r.get("env_runners", {}).get("custom_metrics", {})
        series["unemployment"].append(cm.get("unemployment_rate_mean", np.nan))
        series["inflation"].append(cm.get("inflation_pct_mean", np.nan))
        series["real_gdp"].append(cm.get("gdp_mean", np.nan))  # da la GDP thuc theo pipeline hien tai
        series["gini"].append(cm.get("gini_mean", np.nan))
        series["avg_wage"].append(cm.get("avg_wage_mean", np.nan))
        series["npl"].append(cm.get("bank_npl_mean", np.nan))
        series["active_employees"].append(cm.get("mean_active_employees_mean", np.nan))
    return {k: np.array(v, dtype=float) for k, v in series.items()}


def safe_corr(a, b):
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def diff(x):
    return np.diff(x)


def analyze(series):
    print("\n=== Stylized Facts Validation (Zheng et al. 2020; Stock & Watson 1999) ===\n")

    n = len(series["unemployment"])
    print(f"So diem du lieu (iteration): {n}\n")

    # 1. Phillips Curve (Phillips, 1958; xac nhan lai boi Stock & Watson 1999 qua
    #    cac thanh phan chu ky -- o day dung muc TUYET DOI, don gian hoa vi khong
    #    co du lieu dai han de tach xu huong/chu ky nhu ban goc):
    #    ky vong: tuong quan AM giua unemployment va inflation.
    phillips = safe_corr(series["unemployment"], series["inflation"])
    print(f"[Phillips Curve] corr(unemployment, inflation) = {phillips:.4f}")
    print(f"  Ky vong ly thuyet: AM. {'KHOP' if phillips < -0.1 else ('KHONG KHOP ro rang' if phillips > 0.1 else 'YEU/KHONG RO')}")

    # 2. Okun's Law (Okun, 1963): tuong quan AM giua THAY DOI unemployment va
    #    THAY DOI real GDP.
    d_unemp = diff(series["unemployment"])
    d_gdp = diff(series["real_gdp"])
    okun = safe_corr(d_unemp, d_gdp)
    print(f"\n[Okun's Law] corr(delta_unemployment, delta_real_gdp) = {okun:.4f}")
    print(f"  Ky vong ly thuyet: AM. {'KHOP' if okun < -0.1 else ('KHONG KHOP ro rang' if okun > 0.1 else 'YEU/KHONG RO')}")

    # 3. Wage-unemployment (duong cong Phillips tien luong, cung ho ly thuyet):
    #    ky vong tuong quan AM (that nghiep cao -> ap luc tang luong thap).
    wage_unemp = safe_corr(series["avg_wage"], series["unemployment"])
    print(f"\n[Wage Phillips Curve] corr(avg_wage, unemployment) = {wage_unemp:.4f}")
    print(f"  Ky vong ly thuyet: AM. {'KHOP' if wage_unemp < -0.1 else ('KHONG KHOP ro rang' if wage_unemp > 0.1 else 'YEU/KHONG RO')}")

    # 4. Gini vs unemployment: khong co ky vong dau co dinh chuan trong literature
    #    (phu thuoc Gini do tren THU NHAP hay TAI SAN -- du an nay do tren TAI SAN,
    #    xem rule_engine.py Section 9), chi bao cao de tham khao dinh tinh.
    gini_unemp = safe_corr(series["gini"], series["unemployment"])
    print(f"\n[Gini x Unemployment] corr = {gini_unemp:.4f} (khong co ky vong dau co dinh -- Gini o day do TAI SAN, tham khao)")

    # 5. NPL vs unemployment (ky vong DUONG -- that nghiep cao thuong di kem no
    #    xau tang, phu hop voi tai lieu tin dung-that nghiep da doc: Assenza et al.
    #    2015 tim tuong quan +0.4 giua no va that nghiep tre 5 buoc).
    npl_unemp = safe_corr(series["npl"], series["unemployment"])
    print(f"\n[NPL x Unemployment] corr = {npl_unemp:.4f}")
    print(f"  Ky vong ly thuyet (Assenza et al. 2015): DUONG. {'KHOP' if npl_unemp > 0.1 else ('KHONG KHOP ro rang' if npl_unemp < -0.1 else 'YEU/KHONG RO')}")

    # 6. Phan phoi tai san (Gini): so voi vung tham chieu thuc te da xac minh o
    #    phien truoc (Gini TAI SAN cac nuoc Bac Au ~0.58-0.65, My ~0.85 -- OECD
    #    Wealth Distribution Database) -- CHI mang tinh dinh huong, khong phai
    #    kiem dinh thong ke.
    valid_gini = series["gini"][~np.isnan(series["gini"])]
    if len(valid_gini) > 0:
        print(f"\n[Phan phoi Gini tai san] mean={valid_gini.mean():.3f} min={valid_gini.min():.3f} max={valid_gini.max():.3f}")
        print("  Tham chieu: Bac Au ~0.58-0.65, My ~0.85 (OECD Wealth Distribution DB) -- Gini TAI SAN, khong phai THU NHAP.")

    print("\n[LUU Y PHUONG PHAP] Day la tuong quan tren chuoi ITERATION cua RLlib (moi diem la")
    print("trung binh mot batch nhieu episode-fragment), KHONG phai chuoi thoi gian gia lap lien tuc")
    print("trong 1 the gioi (khac ABIDES-Economist/Assenza et al. dung du lieu he thong that, khong")
    print("qua RLlib training log). Ket qua nen doc la 'xu huong hoi tu cua qua trinh training',")
    print("khong phai 'business cycle facts cua 1 nen kinh te mo phong lien tuc' theo dung nghia")
    print("ban goc Stock & Watson (1999)/Zheng et al. dung tren du lieu test-episode sau khi da hoi tu.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result-json", type=str, required=True)
    args = p.parse_args()

    rows = load_result_json(args.result_json)
    series = extract_series(rows)
    analyze(series)


if __name__ == "__main__":
    main()
