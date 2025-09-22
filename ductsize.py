import os
import math
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
from io import BytesIO
import pdfplumber


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
ROUGHNESS = 0.00015
AIR_DENSITY = 1.2158
AIR_VISCOSITY = 0.0000146
DUCT_TYPE_CIRCULAR = 1
DUCT_TYPE_RECTANGULAR = 2
DUCT_TYPE_OVAL = 3


class DuctRowInput(BaseModel):
    tag: Optional[str] = Field(None, alias="Tag name")
    fitting: Optional[str] = Field("Straight duct", alias="Type fitting")
    flow: float = Field(..., alias="Flow rate, m3/hr")
    length: float = Field(10.0, alias="Length, M/No")
    nomdia: float = Field(400.0, alias="Nom. dia, mm")
    height: float = Field(150.0, alias="Duct height, mm")
    duct_type: Optional[int] = Field(DUCT_TYPE_RECTANGULAR, alias="Duct type")

    class Config:
        allow_population_by_field_name = True


class DuctRowOutput(DuctRowInput):
    width: int = Field(..., alias="Duct width, mm")
    eq_dia: int = Field(..., alias="Eq dia, mm")
    aspect: float = Field(..., alias="Aspect ratio")
    rough: float = Field(..., alias="Roughness in duct")
    reynolds: int = Field(..., alias="Reynolds number in duct")
    friction: float = Field(..., alias="Duct friction co-eff")
    kf: float = Field(..., alias="Fitting K factor PDCF")
    velocity: float = Field(..., alias="Face velocity, m/s")
    pd_pm: float = Field(..., alias="Pressure drop, Pa/m")
    pd: float = Field(..., alias="Pressure drop, Pa")
    critical: str = Field(..., alias="Mark critical path c")

    class Config:
        allow_population_by_field_name = True


fitting_map = {
    "Straight duct": 1,
    "Y joint (L) CFM ratio 0.1": 2,
    "Y joint (L) CFM ratio 0.2": 2.1,
    "Y joint (L) CFM ratio 0.3": 2.2,
    "Y joint (L) CFM ratio 0.4": 2.3,
    "Y joint (L) CFM ratio 0.5": 2.4,
    "Y joint (L) CFM ratio > 0.5": 2.5,
    "Y Joint (B) CFM ratio 0.1": 3,
    "Y Joint (B) CFM ratio 0.2": 3.1,
    "Y Joint (B) CFM ratio 0.3": 3.2,
    "Y Joint (B) CFM ratio 0.4": 3.3,
    "Y Joint (B) CFM ratio > 0.4": 3.4,
    "Reducer, lower size": 4,
    "Diverging, Higher size": 4.1,
    "Offset": 5,
    "Plenum to duct transition": 6,
    "90 deg elbow": 7,
    "45 deg elbow": 7.1,
    "Shoe collar, area ratio < 0.4": 8,
    "Exhaust Louvers": 9,
    "Fresh air Louvers": 10,
    "FSD": 11,
    "Opposed blade damper": 12,
    "Splitter damper": 13,
    "Hit & miss damper": 14,
    "Slot diffusers": 15,
    "Supply air grills": 16,
    "Return air grills": 17,
    "Square diffusers": 18,
    "Circular diffusers": 19,
    "Jet diffusers": 20,
    "VAV till 4100 CFM": 21,
    "VAV above 4100 CFM": 22,
    "VCD 0 degree": 23,
    "VCD 15 degree": 24,
    "VCD 30 degree": 25,
    "VCD 45 degree": 26,
    "VCD 60 degree": 27,
    "Circular damper 0 degree": 28,
    "Circular damper 15 degree": 29,
    "Circular damper 30 degree": 30,
    "Circular damper 45 degree": 31,
    "Perforated grills": 32,
    "Floor grills": 33,
    "Linear supply air grills": 34,
    "Egg crate grills": 35,
    "Swirl diffusers": 36,
    "Plenum VCD, 0 degree": 37,
    "Fan inlet L/H > 5": 38,
    "Flexible duct": 41,
    "Pre-filter": 42,
    "Bag filter": 43,
    "Hepa filter": 44,
    "Coil 1 PD": 45,
    "Coil 2 PD": 46,
    "Coil 3 PD": 47,
    "Coil 4 PD": 48,
    "Coil 5 PD": 49,
    "Miscl 1 PD": 50,
    "Miscl 2 PD": 51,
    "Miscl 3 PD": 52,
}


def friction_coeff(e: float, Re: float, mode: int) -> float:
    if Re > 2320:
        k = 1.325 / (math.log10((e / 3.7) + (5.74 / (Re ** 0.9)))) ** 2
        f = k
        for _ in range(100):
            s = (1.74 - 2 * (math.log10((2 * e) + (18.7 / (Re * (f ** 0.5))))) ) ** 2
            p = 1 / s if s != 0 else 0
            f = p
    else:
        f = 64 / Re
    if mode == 1:
        return f
    elif mode == 2:
        return k
    else:
        return f


def rectangular_duct(de: float, a: float) -> float:
    for b in range(int(a), 10000):
        y = 1.3 * (a * b) ** 0.625 / (a + b) ** 0.25
        if abs(de - y) < 0.1:
            return b
    return a


def oval_duct(de: float, a: float, mode: int) -> float:
    for b in range(int(a) + 1, 10000):
        ar = (math.pi * a ** 2 / 4) + (a * (b - a))
        p = (math.pi * a) + (2 * (b - a))
        y = 1.55 * ar ** 0.625 / p ** 0.25
        if abs(de - y) < 0.1:
            if mode == 1:
                return b
            elif mode == 2:
                return ar
            elif mode == 3:
                return p
    return a


def flexible_duct(D: float, L: float) -> float:
    lef = 1.15 * L
    kc = (lef - L) / L * 100
    pdcf = 1 + (0.58 * kc * math.exp(-0.00496 * D))
    return pdcf


def normalize_col(col: str) -> str:
    return col.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")


def find_tag_col(col_map):
    for col in col_map:
        if "tag" in col and ("name" in col or "tagname" in col or "tag" in col):
            return col_map[col]
    return None


def find_flow_col(col_map):
    for col in col_map:
        norm = normalize_col(col)
        if norm == "fa_cmh" or norm == "fa_(cmh)":
            return col_map[col]
        if norm.replace("_", " ") == "fa (cmh)":
            return col_map[col]
        if "fa (cmh)" in col.lower():
            return col_map[col]
    for col in col_map:
        norm = normalize_col(col)
        if "fa" in norm and "cmh" in norm:
            return col_map[col]
        if "flow" in norm and "m3" in norm:
            return col_map[col]
    return None


class MergedRow(DuctRowInput):
    merged_from: List[int] = []


@app.post("/api/duct/calculate_ducts", response_model=List[DuctRowOutput])
def calculate_ducts(rows: List[DuctRowInput]) -> List[DuctRowOutput]:
    results = []
    for idx, row in enumerate(rows):
        fitting_no = fitting_map.get(row.fitting, 0)

        if row.duct_type == DUCT_TYPE_OVAL:
            eq_dia_m = row.nomdia / 1000
            width = oval_duct(eq_dia_m * 1000, row.height, 1)
            eq_dia = round(eq_dia_m * 1000)
        elif row.duct_type == DUCT_TYPE_RECTANGULAR or row.duct_type is None:
            if row.height > 0:
                if fitting_no < 60:
                    width_unrounded = rectangular_duct(row.nomdia, row.height)
                    width = round(width_unrounded / 50) * 50
                else:
                    width = max(round(row.nomdia * 1.1), int(row.height))
            else:
                width = int(row.nomdia)
            eq_dia_m = 2 * (width / 1000) * (row.height / 1000) / (width / 1000 + row.height / 1000) if row.height > 0 else row.nomdia / 1000
            eq_dia = round(eq_dia_m * 1000)
        else:
            width = int(row.nomdia)
            eq_dia = int(row.nomdia)
            eq_dia_m = row.nomdia / 1000

        aspect_ratio = round(width / row.height, 2) if row.height else 0

        roughness_val = 0.0
        if fitting_no != 0:
            if fitting_no < 2:
                roughness_val = ROUGHNESS / row.nomdia * 1000
            if fitting_no == 41:
                roughness_val += ROUGHNESS / row.nomdia * 1000

        area_m2 = (width / 1000) * (row.height / 1000) if row.height else math.pi * (eq_dia_m / 2) ** 2
        velocity = (row.flow / 3600) / area_m2 if area_m2 else 0

        reynolds_num = 0
        if fitting_no != 0 and velocity > 0:
            reynolds_num = int(round(AIR_DENSITY * velocity * eq_dia_m / AIR_VISCOSITY))

        friction_factor = 0.0
        if fitting_no != 0 and reynolds_num:
            friction_factor = friction_coeff(ROUGHNESS, reynolds_num, 1)

        k_factor = 0.0
        if fitting_no < 2:
            k_factor = 0.0
        elif 2 <= fitting_no < 41:
            k_factor = fitting_no
        elif fitting_no == 41:
            k_factor = flexible_duct(width, row.length)

        face_velocity = 0
        if fitting_no != 0 and fitting_no < 42 and eq_dia_m > 0:
            face_velocity = row.flow * 4 / (3600 * math.pi * (eq_dia_m) ** 2)

        pressure_drop_per_m = 0.0
        if fitting_no != 0 and fitting_no < 60 and velocity > 0:
            pressure_drop_per_m = friction_factor * AIR_DENSITY * velocity ** 2 / (2 * eq_dia_m)

        pressure_drop = 0.0
        D = row.nomdia
        L = row.length
        N = roughness_val
        O = k_factor
        P = pressure_drop_per_m
        V = velocity
        F = fitting_no

        if fitting_no == 0:
            pressure_drop = 0
        else:
            pd1 = (P * V ** 2 * L * N / (2 * D / 1000)) if fitting_no < 2 else 0
            pd2 = (L * (O * V ** 2 / 2 * N)) if 2 <= fitting_no < 41 else 0
            pd3 = (L * F) if fitting_no > 41 else 0
            pd4 = (O * V ** 2 * L * N / (2 * D / 1000)) if fitting_no == 41 else 0
            pressure_drop = pd1 + pd2 + pd3 + pd4

        critical = "c" if pressure_drop_per_m > 0.5 else ""

        results.append(
            DuctRowOutput(
                tag=row.tag,
                fitting=row.fitting,
                flow=row.flow,
                length=row.length,
                nomdia=row.nomdia,
                height=row.height,
                duct_type=row.duct_type,
                width=width,
                eq_dia=eq_dia,
                aspect=aspect_ratio,
                rough=round(roughness_val, 5),
                reynolds=reynolds_num,
                friction=round(friction_factor, 5),
                kf=round(k_factor, 5),
                velocity=round(face_velocity, 5),
                pd_pm=round(pressure_drop_per_m, 5),
                pd=round(pressure_drop, 5),
                critical=critical,
            )
        )
    return results


@app.post("/api/duct/upload_file")
async def upload_file(file: UploadFile = File(...)):
    """Upload CSV, XLSX, or PDF → Detect columns → Calculate ducts → No default row."""
    content = await file.read()
    try:
        if file.filename.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(content))
        elif file.filename.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(BytesIO(content))
        elif file.filename.lower().endswith(".pdf"):
            with pdfplumber.open(BytesIO(content)) as pdf:
                for page in pdf.pages:
                    table = page.extract_table()
                    if table:
                        df = pd.DataFrame(table[1:], columns=[normalize_col(col) for col in table[0]])
                        break
                else:
                    raise HTTPException(status_code=400, detail="No tabular data found in PDF.")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format (CSV/XLSX/PDF only).")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    col_map = {normalize_col(col): col for col in df.columns}
    tag_col = find_tag_col(col_map)
    flow_col = find_flow_col(col_map)

    if not tag_col or not flow_col:
        raise HTTPException(status_code=400, detail="File must contain columns for 'Tag Name' and 'FA (CMH)' (case insensitive)")

    rows = []
    for _, r in df.iterrows():
        try:
            flow_val = r[flow_col]
            flow_val = float(flow_val) if flow_val not in (None, '', 'NA') else 1000.0
            tag_val = str(r[tag_col]) if tag_col in r else ""
            input_obj = DuctRowInput(
                tag=tag_val,
                flow=flow_val
            )
            rows.append(input_obj)
        except Exception:
            continue

    if not rows:
        raise HTTPException(status_code=400, detail="No valid data rows found in file.")

    results = calculate_ducts(rows)
    return {"results": [res.dict(by_alias=True) for res in results]}


@app.post("/api/duct/merge_rows", response_model=MergedRow)
def merge_rows(rows: List[DuctRowInput]):
    merged_row = MergedRow(
        tag="Merged Row",
        fitting="Merged",
        flow=sum(r.flow for r in rows),
        length=sum(r.length for r in rows),
        nomdia=sum(r.nomdia for r in rows),
        height=sum(r.height for r in rows),
        duct_type=rows[0].duct_type if rows else DUCT_TYPE_RECTANGULAR,
        merged_from=[i for i in range(len(rows))],
    )
    return merged_row
