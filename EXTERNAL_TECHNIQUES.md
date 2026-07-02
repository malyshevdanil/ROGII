# Внешние доменные техники (Фаза 4 research-программы)

Источник: **`mycarta/rogii-geosteering-toolkit`** (GitHub) — domain-aware toolkit
специально для соревнования ROGII. Плюс геонавигационная литература (IPTC 2025,
Geophysics 2019). Это первый доменно-обоснованный, внешне-подтверждённый путь
к улучшению — конкретные фичи с отчётами о влиянии.

## Техники с отчётом о влиянии (из ablation-таблицы toolkit'а)

| Техника | Влияние на RMSE | Приоритет |
|---|---|---|
| **Q-3D Tortuosity** | **−0.107** (самая мощная) | ВЫСОКИЙ |
| Multi-scale NCC | (в базе) | — уже в нашем пайплайне |
| Self-correlation (known-секция сама с собой) | + | средний |
| Trajectory features (lateral-only baseline) | + | средний |
| Formation classifier | + | средний |
| Offset-well prior | + | средний |
| Landing-zone state | + | низкий |
| **Sliding distance correlation (dcor)** | (matching) | ВЫСОКИЙ |
| Signed drilling azimuth | (фича/стратификатор) | средний |
| AEON/Catch22 (Catch22+ClaSP) | **+0.476 (ПЕРЕОБУЧАЕТ)** | НЕ ДЕЛАТЬ |

## Ключевые методологические инсайты
- **CV**: StratifiedGroupKFold (стратификация по signed azimuth + median TVT + локация),
  НЕ spatial-block. Причина: валидационные скважины пространственно ПЕРЕМЕШАНЫ с train →
  block-CV неоправданно пессимистичен. (Подтверждает нашу давнюю догадку.)
- **TVT↔Z**: глобально r=−0.96, но ВНУТРИ скважины ≈0 (slope +0.057) → Z сам по себе
  не предсказывает; важна динамика steering. (Совпадает с диагностикой — Simpson's paradox.)
- **Signed azimuth**: бимодальный NW-SE; противоположные азимуты проходят пласты в
  ОБРАТНОМ порядке — важно для сопоставления.
- Модель: единый LightGBM (не PF!) на богатых фичах. Это ДРУГАЯ парадигма, чем наш
  PF+beam+physics; может быть декоррелированным источником для блендинга (Фаза 1).

## Реализации (для переноса)

### Q-3D Tortuosity (wellbore_tortuosity.py)
Вход: MD, X, Y, Z (или MD, inc, azi). Пайплайн:
1. `minimum_curvature()` — 3D траектория (North, East, TVD, DLS) по ISCWSA.
2. `project_to_planes()` — проекция на 2 плоскости: incline (TVD vs гориз. дистанция =
   вертикальное виляние) и azimuth (боковой снос vs гориз. дистанция).
3. `peak_valley_decompose()` — пики/впадины, breakpoints в середине дуги между экстремумами.
4. Per-segment метрики:
   - **T** (tortuosity) = n_segs·Lp·Σ[(lsq/lcq − 1)·(lsq/Lp)] — частотно-взвешенный избыток дуги
   - **Gamma** (deflection) = Lp·(1/m)·Σ tan(θ) — средний угол отклонения между хордами
   - **TQG** = √(T² + Γ²) на плоскость
   - **TQG_Q3D** = √(TQG_incline² + TQG_azimuth²) — итоговый Q-3D скор (Eq.13)
Окна: например 100м MD. Физика: высокая tortuosity = активный steering через пласт =
прокси числа пересечений границ/смены дипа (трудно-предсказуемая часть).

### Sliding distance correlation (dcor_sliding, zone_monitor.py)
```python
def dcor_sliding(series, reference, step=1, normalize=True):
    # slide reference over series, compute distance_correlation at each pos
    # dcor ловит НЕЛИНЕЙНЫЕ зависимости, которые NCC (только линейная) пропускает
    series=np.asarray(series,float); reference=np.asarray(reference,float)
    m=len(reference); n=len(series)
    ref_norm=_zscore(reference) if normalize else reference.copy()
    positions=np.arange(0,n-m+1,step); scores=np.empty(len(positions))
    for k,i in enumerate(positions):
        win=series[i:i+m]; win_norm=_zscore(win) if normalize else win
        scores[k]=dcor.distance_correlation(ref_norm,win_norm)  # pip install dcor
    return scores, positions
```
Наш прошлый NCC-в-PF провалился (MD/TVT рассогласование), но dcor как ФИЧА для GBM
(в TVT-пространстве, не в PF) — не пробовали. Ловит нелинейное сходство формы.

## План интеграции (следующая крупная работа)
1. Реализовать Q-3D tortuosity как per-position фичу (трактабельно, ~день).
2. Добавить signed azimuth, self-correlation, dcor-фичи.
3. Обучить LightGBM с нуля (ROGII_TRAIN_FROM_SCRATCH или свой) с StratifiedGroupKFold.
4. Ablation: подтвердить, что каждая фича зарабатывает место (как в toolkit'е).
5. Блендинг этой GBM-модели (другая парадигма) с нашим PF+physics — потенциал декорреляции.

Ссылки:
- https://github.com/mycarta/rogii-geosteering-toolkit
- IPTC 2025 "Multi-Scale Cross-Correlation ... Well Log Depth Matching"
- Geophysics 2019 "Stochastic clustering and pattern matching for real-time geosteering"
