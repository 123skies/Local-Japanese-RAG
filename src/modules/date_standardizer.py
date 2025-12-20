# project_root/src/modules/date_standardizer.py
import re

class DateStandardizer:
    def __init__(self):
        # 和暦定義 (元号名, 開始西暦年)
        self.era_list = [
            ("令和", 2019), ("平成", 1989), ("昭和", 1926), ("大正", 1912), ("明治", 1868),
            ("慶応", 1865), ("元治", 1864), ("文久", 1861), ("万延", 1860), ("安政", 1854),
            ("嘉永", 1848), ("弘化", 1844), ("天保", 1830), ("文政", 1818), ("文化", 1804),
            ("享和", 1801), ("寛政", 1789), ("天明", 1781), ("安永", 1772), ("明和", 1764),
            ("宝暦", 1751), ("寛延", 1748), ("延享", 1744), ("寛保", 1741), ("元文", 1736),
            ("享保", 1716), ("正徳", 1711), ("宝永", 1704), ("元禄", 1688), ("貞享", 1684),
            ("天和", 1681), ("延宝", 1673), ("寛文", 1661), ("万治", 1658), ("明暦", 1655),
            ("承応", 1652), ("慶安", 1648), ("正保", 1644), ("寛永", 1624), ("元和", 1615),
            ("慶長", 1596)
        ]
        
        self.era_names = "|".join([e[0] for e in self.era_list])
        
        # 数字パターン：アラビア数字、漢数字、全角数字、「元」
        self.num_pat = r'[0-9０-９一二三四五六七八九十百〇零]+|元'
        
        self.kanji_map = str.maketrans('０１２３４５６７８９', '0123456789')
        self.kanji_nums = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, 
                           '六': 6, '七': 7, '八': 8, '九': 9}

        # --- 正規表現パターンの構築 ---
        
        # 1. 和暦(西暦) パターン -> 和暦を正とする
        # 例: 昭和26年(1941), 昭和二六年(一九四一)
        self.pat_era_master = re.compile(
            rf'(?P<era_m>{self.era_names})\s*(?P<enum_m>{self.num_pat})\s*年?\s*[（\(].*?(?P<west_sub>{self.num_pat})\s*年?.*?[）\)]\s*年?'
        )

        # 2. 西暦(和暦) パターン -> 西暦を正とする
        # 例: 1951年(昭和20年), 一九一一(明治四四)年
        self.pat_west_master = re.compile(
            rf'(?P<west_m>{self.num_pat})\s*年?\s*[（\(].*?(?P<era_sub>{self.era_names})\s*(?P<enum_sub>{self.num_pat})\s*年?.*?[）\)]\s*年?'
        )

        # 3. 和暦のみ
        # 「年」必須により範囲外の漢数字を無視
        self.pat_era_only = re.compile(
            rf'(?P<era_o>{self.era_names})\s*(?P<enum_o>{self.num_pat})\s*年'
        )

        # 4. 西暦のみ
        # 「年」必須により、単なる4桁の数字（電話番号や番地）を無視
        self.pat_west_only = re.compile(
            rf'(?P<west_o>{self.num_pat})\s*年'
        )

    def _normalize_number(self, text):
        """数字文字列を整数に変換"""
        if not text: return 0
        if text == "元": return 1
        
        # 1. 全角アラビア数字を半角に
        text = text.translate(self.kanji_map)
        
        # 2. その状態で全て数字なら即int化
        if text.isdigit(): return int(text)
        
        # 3. 漢数字の処理
        if '十' in text or '百' in text:
            # 位取り記法（二十四 など）
            total = 0
            current = 0
            for char in text:
                if char in self.kanji_nums:
                    current = self.kanji_nums[char]
                elif char == '十':
                    total += (current if current > 0 else 1) * 10
                    current = 0
                elif char == '百':
                    total += (current if current > 0 else 1) * 100
                    current = 0
                elif char.isdigit():
                    current = int(char)
            total += current
            return total
        else:
            # 位取りなし（一九一一、四四 など）
            temp_str = ""
            for char in text:
                if char in self.kanji_nums:
                    temp_str += str(self.kanji_nums[char])
                elif char.isdigit():
                    temp_str += char
                elif char == '〇' or char == '零':
                    temp_str += '0'
            
            if temp_str:
                return int(temp_str)
            return 0

    def _to_western(self, era_name, era_year):
        """和暦名と年数から西暦年を算出"""
        for name, start_year in self.era_list:
            if name == era_name:
                return start_year + era_year - 1
        return None

    def _to_wareki(self, western_year):
        """西暦年から和暦（元号, 年数）を算出"""
        for name, start_year in self.era_list:
            if western_year >= start_year:
                era_year = western_year - start_year + 1
                return name, era_year
        return None, None

    def _format_output(self, era_name, era_year, western_year):
        """統一フォーマット：和暦XX年（西暦YYYY年）"""
        era_str = "元年" if era_year == 1 else f"{era_year}年"
        # return f"{western_year}年（{era_name}{era_str}）" # 出力形式：西暦（和暦）
        return f"{era_name}{era_str}（{western_year}年）" # 出力形式：和暦（西暦）

    def process_text(self, text):
        """テキスト全体に対して置換処理を行う"""
        
        def replacer(match):
            m = match.groupdict()
            original_text = match.group(0)

            target_west = 0
            target_era_name = ""
            target_era_year = 0

            # 1. 和暦マスター
            if 'era_m' in m and m['era_m']:
                target_era_name = m['era_m']
                target_era_year = self._normalize_number(m['enum_m'])
                target_west = self._to_western(target_era_name, target_era_year)
            
            # 2. 西暦マスター
            elif 'west_m' in m and m['west_m']:
                target_west = self._normalize_number(m['west_m'])
                target_era_name, target_era_year = self._to_wareki(target_west)
            
            # 3. 和暦のみ
            elif 'era_o' in m and m['era_o']:
                target_era_name = m['era_o']
                target_era_year = self._normalize_number(m['enum_o'])
                target_west = self._to_western(target_era_name, target_era_year)
            
            # 4. 西暦のみ
            elif 'west_o' in m and m['west_o']:
                target_west = self._normalize_number(m['west_o'])
                target_era_name, target_era_year = self._to_wareki(target_west)

            # 変換成功なら統一フォーマットで返す
            if target_west and target_era_name:
                return self._format_output(target_era_name, target_era_year, target_west)
            
            # 変換不可（範囲外など）なら元のテキストを返す
            return original_text

        combined_pattern = re.compile(
            f"{self.pat_era_master.pattern}|{self.pat_west_master.pattern}|{self.pat_era_only.pattern}|{self.pat_west_only.pattern}"
        )
        
        return combined_pattern.sub(replacer, text)