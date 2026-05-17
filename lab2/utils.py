import numpy as np
import pandas as pd


def decode_win(text):
    return text.encode('latin1').decode('cp1251')


def get_normal_city(city) -> str:
    """Ультразахардкоженно!! Позже можно будет переписать на что-то адекватное"""
    # [   'БЛАГОВЕЩЕНСК',     'Благовещенс',    'Благовещенск',   'Благовещенскк', 'благовещенск'
    #        'ГЕЛЕНДЖИК',        'Геленджи',       'Геленджик',      'Геленджикк', 'геленджик'
    #           'Москва',         'Находка', 'Санкт-Петербург',
    #           'Сочи', 'Сычи',    ,       ]

    city = city.lower()
    if "благо" in city:
        return "Благовещенск"
    elif "гелен" in city:
        return "Геленджик"
    elif "москва" in city:
        return "Москва"
    elif "находка" in city:
        return "Находка"
    elif "санкт-петербург" in city:
        return "Санкт-Петербург"
    elif "с" in city:
        return "Сочи"
    else:
        return "втф"


def get_clean_df(df):
    # Фикс кодировки city
    df['city'] = df['city'].apply(lambda x: get_normal_city(decode_win(x)) if isinstance(x, str) else x)

    # Все колонки, кроме даты и города – числа
    numeric_cols = [c for c in df.columns if c not in ['ds', 'city']]
    for col in numeric_cols:
        # Заменяем запятые на точки (6,7 -> 6.7)
        df[col] = df[col].astype(str).str.replace(',', '.')
        # errors='coerce': превратит всякую лабуду в np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ниже находим и удаляем строки без дат
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df = df.dropna(subset=['ds'])

    # Удаляем unnamed и полностью пустые колонки
    df = df.dropna(axis=1, how='all')
    df = df.loc[:, ~df.columns.str.contains('unnamed', case=False)]

    return df


def interpolate_time_by_group(group):
    # Сохраняем city
    city_name = group.name

    group = group.set_index('ds')

    num_cols = group.select_dtypes(include=[np.number]).columns
    # Интерполяция каждый час
    interpolated_group = group[num_cols].resample('1h').mean().interpolate(method='time')
    # corner-cases
    interpolated_group = interpolated_group.bfill().ffill()

    # Возвращаем city ко всем строкам (и новым тоже)
    interpolated_group['city'] = city_name
    return interpolated_group.reset_index()
