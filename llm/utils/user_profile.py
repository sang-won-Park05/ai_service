# ai_service/llm/utils/user_profile.py
from __future__ import annotations

import os
from typing import Any, Dict, List

import psycopg2
from psycopg2.extras import RealDictCursor

from ..config import settings  # 이미 있는 config 재사용


def get_db_conn():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "medinote"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
    )
    return conn


def load_user_profile(user_id: int) -> Dict[str, Any]:
    """
    user_id 기준으로:
      - app_user (기본 정보)
      - user_chronic_disease (만성질환)
      - user_allergy (알레르기)
    를 한 번에 읽어와 dict로 반환.
    """
    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 1) 기본 정보
            cur.execute(
                """
                SELECT
                    user_id,
                    date_of_birth,
                    sex,
                    blood_type,
                    height_cm,
                    weight_kg,
                    drinking_status,
                    smoking_status
                FROM app_user
                WHERE user_id = %s
                """,
                (user_id,),
            )
            basic = cur.fetchone()

            # 2) 만성질환
            cur.execute(
                """
                SELECT
                    disease_name,
                    disease_type,
                    main_medication,
                    diagnosed_at,
                    is_active,
                    memo
                FROM user_chronic_disease
                WHERE user_id = %s
                ORDER BY disease_name
                """,
                (user_id,),
            )
            chronic = cur.fetchall()

            # 3) 알레르기
            cur.execute(
                """
                SELECT
                    allergen_name,
                    allergy_type,
                    severity,
                    reaction,
                    memo
                FROM user_allergy
                WHERE user_id = %s
                ORDER BY allergen_name
                """,
                (user_id,),
            )
            allergies = cur.fetchall()

        return {
            "user_id": user_id,
            "basic": basic,
            "chronic_diseases": chronic,
            "allergies": allergies,
        }
    finally:
        conn.close()
