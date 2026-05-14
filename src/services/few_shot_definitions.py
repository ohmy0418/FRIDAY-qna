# Few-shot prompt assets — single source of truth for runtime (`qna_few_shot`).
data = {'tasks': {'db_user_answer_style': {'description': '질문에 맞춰 한두 문장으로만 답하는 스타일',
                                    'examples': [{'id': 'answer_email_only',
                                                  'messages': [{'content': '아래 [참고 정보]만 사용해, 사용자 질문에 직접 답하세요. 불필요한 항목은 '
                                                                           '말하지 마세요.\n'
                                                                           '\n'
                                                                           '[참고 정보]\n'
                                                                           '이름: 문영철 / 이메일: menjkl@mobigen.com / 부서: '
                                                                           'DX개발1팀\n'
                                                                           '\n'
                                                                           '사용자 질문:\n'
                                                                           '문영철 프로의 이메일 주소를 알려줘.\n',
                                                                'role': 'user'},
                                                               {'content': '문영철 프로의 이메일 주소는 menjkl@mobigen.com 입니다.',
                                                                'role': 'assistant'}]},
                                                 {'id': 'answer_dept_only',
                                                  'messages': [{'content': '아래 [참고 정보]만 사용해, 사용자 질문에 직접 답하세요. 불필요한 항목은 '
                                                                           '말하지 마세요.\n'
                                                                           '\n'
                                                                           '[참고 정보]\n'
                                                                           '이름: 문영철 / 이메일: menjkl@mobigen.com / 부서: '
                                                                           'DX개발1팀\n'
                                                                           '\n'
                                                                           '사용자 질문:\n'
                                                                           '문영철 프로의 담당 부서를 알려줘.\n',
                                                                'role': 'user'},
                                                               {'content': '문영철 프로의 담당 부서는 DX개발1팀 입니다.',
                                                                'role': 'assistant'}]}]},
           'employee_dept_and_name_lookup': {'description': '질문 안에서 사원명(보통 "OOO 프로")과 조직·부서명을 찾아 한 쌍으로 묶는다. 문장 앞에 조직이 '
                                                            '올 수도, 사원명이 먼저 올 수도 있으며, 질문에 이메일 주소가 같이 나와도 (확인·맞는지 등) 동일 '
                                                            '인물 조회로 본다. 사용자가 묻은 항목만 답하고, 조직명만 단독 검색어처럼 오인해 실패 안내만 하지 '
                                                            '않는다.\n',
                                             'examples': [{'id': 'dept_then_name_in_email_kim',
                                                           'messages': [{'content': '규칙:\n'
                                                                                    '- 문장에 조직·부서명과 "OOO 프로"(사원명)가 함께 '
                                                                                    '있으면, 순서와 관계없이 그 사원을 조직 조건으로 좁혀 '
                                                                                    '조회한 결과로 답한다.\n'
                                                                                    '- "A인 B 프로" / "A의 B 프로"에서 A는 '
                                                                                    '필터(소속), B는 사원명. 조사 "인"은 답에서는 "의" '
                                                                                    '등 자연스러운 표현으로 바꿔도 된다.\n'
                                                                                    '- 이메일만 물었으면 이메일만 답한다.\n'
                                                                                    '\n'
                                                                                    '[조회 결과 / 참고]\n'
                                                                                    '소속(또는 부서 단위): 메시징사업그룹\n'
                                                                                    '사원명: 김형근\n'
                                                                                    '이메일: nomota@mobigen.com\n'
                                                                                    '\n'
                                                                                    '사용자 질문:\n'
                                                                                    '메시징사업그룹인 김형근 프로의 이메일 주소를 알려줘\n',
                                                                         'role': 'user'},
                                                                        {'content': '메시징사업그룹의 김형근 프로의 이메일 주소는 '
                                                                                    'nomota@mobigen.com입니다.',
                                                                         'role': 'assistant'}]},
                                                          {'id': 'name_then_dept_email_kim',
                                                           'messages': [{'content': '규칙:\n'
                                                                                    '- 사원명이 앞에 오고 뒤에 소속·부서가 나와도 동일하게 한 '
                                                                                    '사람으로 해석한다.\n'
                                                                                    '\n'
                                                                                    '[조회 결과 / 참고]\n'
                                                                                    '소속(또는 부서 단위): 메시징사업그룹\n'
                                                                                    '사원명: 김형근\n'
                                                                                    '이메일: nomota@mobigen.com\n'
                                                                                    '\n'
                                                                                    '사용자 질문:\n'
                                                                                    '김형근 프로 메시징사업그룹 소속 맞지? 이메일 주소 좀 '
                                                                                    '알려줘\n',
                                                                         'role': 'user'},
                                                                        {'content': '메시징사업그룹 소속 김형근 프로의 이메일 주소는 '
                                                                                    'nomota@mobigen.com입니다.',
                                                                         'role': 'assistant'}]},
                                                          {'id': 'name_dept_compact_email_kim',
                                                           'messages': [{'content': '규칙:\n'
                                                                                    '- "소속", "부서", "팀" 등 접속사 없이 나열되어도 '
                                                                                    '맥락상 조직명+사원명이면 같은 방식으로 해석한다.\n'
                                                                                    '\n'
                                                                                    '[조회 결과 / 참고]\n'
                                                                                    '소속(또는 부서 단위): 메시징사업그룹\n'
                                                                                    '사원명: 김형근\n'
                                                                                    '이메일: nomota@mobigen.com\n'
                                                                                    '\n'
                                                                                    '사용자 질문:\n'
                                                                                    '메시징사업그룹 김형근 프로 이메일\n',
                                                                         'role': 'user'},
                                                                        {'content': '메시징사업그룹의 김형근 프로의 이메일 주소는 '
                                                                                    'nomota@mobigen.com입니다.',
                                                                         'role': 'assistant'}]},
                                                          {'id': 'name_with_email_in_question_confirm',
                                                           'messages': [{'content': '규칙:\n'
                                                                                    '- 질문에 이메일 문자열이 포함되어 있어도, 사원명·소속과 '
                                                                                    '함께 묻는 경우에는 "등록 정보와 일치 여부" 또는 해당 '
                                                                                    '사원의 이메일 안내로 답한다.\n'
                                                                                    '- 확인 질문이면 짧게 맞음/틀림과 등록 이메일을 말한다.\n'
                                                                                    '\n'
                                                                                    '[조회 결과 / 참고]\n'
                                                                                    '소속(또는 부서 단위): 메시징사업그룹\n'
                                                                                    '사원명: 김형근\n'
                                                                                    '이메일: nomota@mobigen.com\n'
                                                                                    '\n'
                                                                                    '사용자 질문:\n'
                                                                                    '메시징사업그룹 김형근 프로 이메일 '
                                                                                    'nomota@mobigen.com 맞아?\n',
                                                                         'role': 'user'},
                                                                        {'content': '네, 메시징사업그룹 소속 김형근 프로의 이메일은 '
                                                                                    'nomota@mobigen.com으로 등록되어 있습니다.',
                                                                         'role': 'assistant'}]}]},
           'policy_condition_calculation': {'description': '사용자가 직책·거리·금액·일수·연료 종류 등 구체적인 조건을 제시했을 때, 문서의 기준표를 적용해 '
                                                           '계산하고 산식을 굵게 명시한다. 문서에 없는 값은 절대 추측하지 않는다.\n',
                                            'examples': [{'id': 'fuel_cost_distance_calculation',
                                                          'messages': [{'content': '아래 [문서 내용]만 근거로 질문에 답하세요.\n'
                                                                                   '사용자가 거리·연료 종류 등 조건을 제시한 경우 문서 기준을 '
                                                                                   '적용해 계산하고,\n'
                                                                                   '산식을 **굵게** 명시하세요.\n'
                                                                                   '\n'
                                                                                   '[문서 내용]\n'
                                                                                   '[개인차량 유류비 지급기준]\n'
                                                                                   '업무 목적 개인차량 사용 시 연료 종류별 km당 지원 단가:\n'
                                                                                   '- 휘발유: 250원/km\n'
                                                                                   '- 경유: 200원/km\n'
                                                                                   '- LPG: 150원/km\n'
                                                                                   '\n'
                                                                                   '사용자 질문:\n'
                                                                                   '외근 나가서 경유로 왕복 100km 다녀왔어. 얼마 지원받을 '
                                                                                   '수 있어?\n',
                                                                        'role': 'user'},
                                                                       {'content': "'개인차량 유류비 지급기준'에 따르면 **20,000원**을 "
                                                                                   '지원받을 수 있습니다.\n'
                                                                                   '\n'
                                                                                   '**100km × 200원/km = 20,000원**\n',
                                                                        'role': 'assistant'}]},
                                                         {'id': 'equipment_budget_position_role_calculation',
                                                          'messages': [{'content': '아래 [문서 내용]만 근거로 질문에 답하세요.\n'
                                                                                   '사용자가 직책·직무 등 조건을 제시한 경우 해당 행을 찾아 '
                                                                                   '합산하고,\n'
                                                                                   '계산 내역을 **굵게** 명시하세요.\n'
                                                                                   '\n'
                                                                                   '[문서 내용]\n'
                                                                                   '[업무용 장비 지급 및 관리에 관한 지침]\n'
                                                                                   '직책별 기준 금액:\n'
                                                                                   '- 사원/대리: 1,200,000원\n'
                                                                                   '- 과장/차장: 1,500,000원\n'
                                                                                   '- 책임/부장: 1,700,000원\n'
                                                                                   '직무별 추가 금액:\n'
                                                                                   '- 디자인: +200,000원\n'
                                                                                   '- 개발: +300,000원\n'
                                                                                   '\n'
                                                                                   '사용자 질문:\n'
                                                                                   '내가 책임이고 디자인 파트인데, 장비 구매 기준 금액이 '
                                                                                   '얼마야?\n',
                                                                        'role': 'user'},
                                                                       {'content': "'업무용 장비 지급 및 관리에 관한 지침'에 따르면 총 "
                                                                                   '**1,900,000원**입니다.\n'
                                                                                   '\n'
                                                                                   '**1,700,000원(책임 기준) + 200,000원(디자인 '
                                                                                   '추가) = 1,900,000원**\n',
                                                                        'role': 'assistant'}]}]},
           'regulation_owner_paraphrase': {'description': '"어느 부서가 담당이야" / "담당부서는 어디야" / "관련 규정 … 담당" 등은 같은 의도. 등록부에 '
                                                          '담당 부서가 있으면 짧게 동일 포맷으로 답한다(한 가지 실패 문구로 끝내지 않음).\n',
                                           'examples': [{'id': 'leave_policy_owner_which_dept',
                                                         'messages': [{'content': '규칙:\n'
                                                                                  '- 담당 부서를 묻는 표현이 문장마다 달라도, 확인된 담당 '
                                                                                  '부서가 있으면 같은 형식으로 답한다.\n'
                                                                                  '- 형식 예: "담당 부서는 OOO입니다."\n'
                                                                                  '\n'
                                                                                  '[참고: 사내 규정 업무 등록부 조회 결과]\n'
                                                                                  '주제: 휴가 관련 규정\n'
                                                                                  '담당 부서: 피플팀\n'
                                                                                  '\n'
                                                                                  '사용자 질문:\n'
                                                                                  '휴가 규정은 어느 부서가 담당이야?\n',
                                                                       'role': 'user'},
                                                                      {'content': '담당 부서는 피플팀입니다.',
                                                                       'role': 'assistant'}]},
                                                        {'id': 'leave_policy_owner_where_dept',
                                                         'messages': [{'content': '규칙:\n'
                                                                                  '- 담당 부서를 묻는 표현이 문장마다 달라도, 확인된 담당 '
                                                                                  '부서가 있으면 같은 형식으로 답한다.\n'
                                                                                  '\n'
                                                                                  '[참고: 사내 규정 업무 등록부 조회 결과]\n'
                                                                                  '주제: 휴가 관련 규정\n'
                                                                                  '담당 부서: 피플팀\n'
                                                                                  '\n'
                                                                                  '사용자 질문:\n'
                                                                                  '휴가 규정 담당부서는 어디야?\n',
                                                                       'role': 'user'},
                                                                      {'content': '담당 부서는 피플팀입니다.',
                                                                       'role': 'assistant'}]},
                                                        {'id': 'leave_policy_owner_related_regulation',
                                                         'messages': [{'content': '규칙:\n'
                                                                                  '- 담당 부서를 묻는 표현이 문장마다 달라도, 확인된 담당 '
                                                                                  '부서가 있으면 같은 형식으로 답한다.\n'
                                                                                  '\n'
                                                                                  '[참고: 사내 규정 업무 등록부 조회 결과]\n'
                                                                                  '주제: 휴가 관련 규정\n'
                                                                                  '담당 부서: 피플팀\n'
                                                                                  '\n'
                                                                                  '사용자 질문:\n'
                                                                                  '휴가 관련한 규정은 어느 부서가 담당이야?\n',
                                                                       'role': 'user'},
                                                                      {'content': '담당 부서는 피플팀입니다.',
                                                                       'role': 'assistant'}]}]},
           'route_classification': {'description': '질문 → route_type JSON 한 개',
                                    'examples': [{'id': 'route_db_employee_email',
                                                  'messages': [{'content': '당신은 QnA 라우팅 분류기입니다.\n'
                                                                           '사용자 질문을 route_type 중 하나로 분류하고, 반드시 JSON만 '
                                                                           '출력합니다.\n'
                                                                           '형식: '
                                                                           '{"route_type":"rag|db_api|hybrid|unsupported","reason":"짧은 '
                                                                           '근거"}\n'
                                                                           '\n'
                                                                           '사용자 질문:\n'
                                                                           '"""문영철 프로의 이메일 주소를 알려줘."""\n',
                                                                'role': 'user'},
                                                               {'content': '{"route_type":"db_api","reason":"사원 '
                                                                           '연락처(이메일) 정형 DB 조회"}',
                                                                'role': 'assistant'}]},
                                                 {'id': 'route_rag_policy_only',
                                                  'messages': [{'content': '당신은 QnA 라우팅 분류기입니다.\n'
                                                                           '사용자 질문을 route_type 중 하나로 분류하고, 반드시 JSON만 '
                                                                           '출력합니다.\n'
                                                                           '형식: '
                                                                           '{"route_type":"rag|db_api|hybrid|unsupported","reason":"짧은 '
                                                                           '근거"}\n'
                                                                           '\n'
                                                                           '사용자 질문:\n'
                                                                           '"""법인카드 사용 규정에서 영수증 제출 기한이 어떻게 돼?"""\n',
                                                                'role': 'user'},
                                                               {'content': '{"route_type":"rag","reason":"사내 규정/문서 기반 '
                                                                           '비정형 검색"}',
                                                                'role': 'assistant'}]},
                                                 {'id': 'route_hybrid_both',
                                                  'messages': [{'content': '당신은 QnA 라우팅 분류기입니다.\n'
                                                                           '사용자 질문을 route_type 중 하나로 분류하고, 반드시 JSON만 '
                                                                           '출력합니다.\n'
                                                                           '형식: '
                                                                           '{"route_type":"rag|db_api|hybrid|unsupported","reason":"짧은 '
                                                                           '근거"}\n'
                                                                           '\n'
                                                                           '사용자 질문:\n'
                                                                           '"""출장비 정산 절차 알려주고, 담당 부서 연락처도 알려줘."""\n',
                                                                'role': 'user'},
                                                               {'content': '{"route_type":"hybrid","reason":"규정 문서와 담당 '
                                                                           '부서 정형 정보가 모두 필요"}',
                                                                'role': 'assistant'}]},
                                                 {'id': 'route_db_leave_policy_owner_dept_v1',
                                                  'messages': [{'content': '당신은 QnA 라우팅 분류기입니다.\n'
                                                                           '사용자 질문을 route_type 중 하나로 분류하고, 반드시 JSON만 '
                                                                           '출력합니다.\n'
                                                                           '형식: '
                                                                           '{"route_type":"rag|db_api|hybrid|unsupported","reason":"짧은 '
                                                                           '근거"}\n'
                                                                           '\n'
                                                                           '사용자 질문:\n'
                                                                           '"""휴가 규정은 어느 부서가 담당이야?"""\n',
                                                                'role': 'user'},
                                                               {'content': '{"route_type":"db_api","reason":"사내 규정 업무의 '
                                                                           '담당 부서(정형 등록부) 조회"}',
                                                                'role': 'assistant'}]},
                                                 {'id': 'route_db_leave_policy_owner_dept_v2',
                                                  'messages': [{'content': '당신은 QnA 라우팅 분류기입니다.\n'
                                                                           '사용자 질문을 route_type 중 하나로 분류하고, 반드시 JSON만 '
                                                                           '출력합니다.\n'
                                                                           '형식: '
                                                                           '{"route_type":"rag|db_api|hybrid|unsupported","reason":"짧은 '
                                                                           '근거"}\n'
                                                                           '\n'
                                                                           '사용자 질문:\n'
                                                                           '"""휴가 규정 담당부서는 어디야?"""\n',
                                                                'role': 'user'},
                                                               {'content': '{"route_type":"db_api","reason":"사내 규정 업무의 '
                                                                           '담당 부서(정형 등록부) 조회"}',
                                                                'role': 'assistant'}]},
                                                 {'id': 'route_rag_position_condition_equipment',
                                                  'messages': [{'content': '당신은 QnA 라우팅 분류기입니다.\n'
                                                                           '사용자 질문을 route_type 중 하나로 분류하고, 반드시 JSON만 '
                                                                           '출력합니다.\n'
                                                                           '형식: '
                                                                           '{"route_type":"rag|db_api|hybrid|unsupported","reason":"짧은 '
                                                                           '근거"}\n'
                                                                           '\n'
                                                                           '사용자 질문:\n'
                                                                           '"""내 직책이 책임이면 장비 기준금액이 얼마야?"""\n',
                                                                'role': 'user'},
                                                               {'content': '{"route_type":"rag","reason":"사용자가 직책 조건을 '
                                                                           '직접 명시했으므로 문서 기준표 조회만 필요"}',
                                                                'role': 'assistant'}]},
                                                 {'id': 'route_rag_position_condition_expense',
                                                  'messages': [{'content': '당신은 QnA 라우팅 분류기입니다.\n'
                                                                           '사용자 질문을 route_type 중 하나로 분류하고, 반드시 JSON만 '
                                                                           '출력합니다.\n'
                                                                           '형식: '
                                                                           '{"route_type":"rag|db_api|hybrid|unsupported","reason":"짧은 '
                                                                           '근거"}\n'
                                                                           '\n'
                                                                           '사용자 질문:\n'
                                                                           '"""직책이 과장이면 출장비 상한이 얼마야?"""\n',
                                                                'role': 'user'},
                                                               {'content': '{"route_type":"rag","reason":"사용자가 직책 조건을 '
                                                                           '직접 명시했으므로 문서 기준표 조회만 필요"}',
                                                                'role': 'assistant'}]}]},
           'structured_db_input_normalization': {'description': 'Case 4: 이름과 직함이 붙어 있으면(예: 이미선프로) 직함 앞에 공백을 넣어 사원명·직함을 '
                                                                '분리한다. Case 5: 부서명 띄어쓰기 변형(예: 지능 데이터 팀 ↔ 지능데이터팀)은 '
                                                                'compact 매칭·별칭·유사 canonical 매핑으로 DB ILIKE 전에 정리한다. 답변 '
                                                                '시 사용자 표기를 그대로 두어도 되지만, 조회 근거는 정규화된 값을 따른다.\n',
                                                 'examples': [{'id': 'case4_glued_honorific_then_lookup',
                                                               'messages': [{'content': '[참고 — 서버 정규화 후 조회됨]\n'
                                                                                        '원문에 "이미선프로"처럼 붙어 있어도 조회 파라미터는 '
                                                                                        '"이미선" + 직함 "프로"로 분리된다.\n'
                                                                                        '\n'
                                                                                        '[조회 결과]\n'
                                                                                        '사원명: 이미선 / 부서: 피플팀 / 휴대폰: '
                                                                                        '(등록값)\n'
                                                                                        '\n'
                                                                                        '사용자 질문:\n'
                                                                                        '피플팀 이미선프로 휴대폰 번호 알려줘\n',
                                                                             'role': 'user'},
                                                                            {'content': '피플팀 소속 이미선 프로의 휴대폰 번호는 '
                                                                                        '(등록값)입니다.',
                                                                             'role': 'assistant'}]},
                                                              {'id': 'case5_dept_space_variant',
                                                               'messages': [{'content': '[참고 — 서버 정규화]\n'
                                                                                        '"지능 데이터 팀"처럼 공백이 끼어 있어도 '
                                                                                        'canonical 부서명·compact ILIKE로 '
                                                                                        '매핑된다.\n'
                                                                                        '\n'
                                                                                        '[조회 결과]\n'
                                                                                        '부서: 지능데이터 서비스팀 / 인원 또는 명단: '
                                                                                        '(등록값)\n'
                                                                                        '\n'
                                                                                        '사용자 질문:\n'
                                                                                        '지능 데이터 팀은 몇 명이야?\n',
                                                                             'role': 'user'},
                                                                            {'content': '지능데이터 서비스팀은 (등록값)명으로 구성되어 '
                                                                                        '있습니다.',
                                                                             'role': 'assistant'}]}]}},
 'version': 1}
