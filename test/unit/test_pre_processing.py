import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal
from pre_processing import Preparing
import joblib


class TestPreparing:
    _EXPECTED_DF_COLUMNS = ['index', 'groups', 'when', 'super_hero_group', 'tracking', 'place', 'tracking_times', 'crystal_type', 'Unnamed: 7',
                            'human_behavior_report', 'human_measure', 'crystal_weight', 'expected_factor_x', 'previous_factor_x', 'first_factor_x',
                            'expected_final_factor_x', 'final_factor_x', 'previous_adamantium', 'Unnamed: 17', 'etherium_before_start', 'opened',
                            'chemical_x', 'raw_kryptonite', 'argon', 'pure_seastone', 'crystal_supergroup', 'Cycle', 'diff_process_end', 'diff_subprocess1',
                            'diff_process', 'diff_on_tower']
    _EXPECTED_DF_RAW_COLUMNS = ['index', 'groups', 'when', 'super_hero_group', 'tracking', 'place', 'tracking_times', 'crystal_type', 'Unnamed: 7',
                                'human_behavior_report', 'human_measure', 'crystal_weight', 'expected_factor_x', 'previous_factor_x', 'first_factor_x',
                                'expected_final_factor_x', 'final_factor_x', 'previous_adamantium', 'Unnamed: 17', 'etherium_before_start', 'expected_start',
                                'start_process', 'start_subprocess1', 'start_critical_subprocess1', 'predicted_process_end', 'process_end', 'subprocess1_end',
                                'reported_on_tower', 'opened', 'chemical_x', 'raw_kryptonite', 'argon', 'pure_seastone', 'crystal_supergroup', 'Cycle']
    _EXPECTED_DF_TARGET_NAME = 'target'
    _EXPECTED_DF = joblib.load('test/unit/test_files/df.pkl')
    _EXPECTED_DF_RAW = joblib.load('test/unit/test_files/df_raw.pkl')
    _EXPECTED_DF_TARGET = joblib.load('test/unit/test_files/df_target.pkl')

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.parametrize("expected_df_columns, expected_df_raw_columns, expected_df_target_name, expected_df, expected_df_raw, expected_target",
                             [(_EXPECTED_DF_COLUMNS, _EXPECTED_DF_RAW_COLUMNS, _EXPECTED_DF_TARGET_NAME, _EXPECTED_DF, _EXPECTED_DF_RAW, _EXPECTED_DF_TARGET)])
    def test_getsValidInput_shouldReturnCorrectDataFrames(self, expected_df_columns: list, expected_df_raw_columns: list, expected_df_target_name: str,
                                                          expected_df: pd.DataFrame, expected_df_raw: pd.DataFrame, expected_target: pd.Series) -> None:
        df, df_raw, df_target = Preparing.data_prepared()

        assert list(df.columns) == expected_df_columns
        assert list(df_raw.columns) == expected_df_raw_columns
        assert df_target.name == expected_df_target_name

        assert_frame_equal(df, expected_df)
        assert_frame_equal(df_raw, expected_df_raw)
        assert_series_equal(df_target, expected_target)
