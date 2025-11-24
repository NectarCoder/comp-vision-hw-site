import os

import cv2
import numpy as np


from hwsources.module2_part1 import match_template


def _make_test_images(tmp_path):
    # White background scene
    scene = np.full((150, 150, 3), 255, dtype=np.uint8)
    # Template with internal pattern (small dark square inside white patch)
    template = np.full((20, 20, 3), 255, dtype=np.uint8)
    template[8:12, 8:12] = 0

    # Place a clean (strong) template copy at (30,20)
    scene[20:40, 30:50] = template

    # Place a weaker (faded) version of the template at (90,80)
    faded = (template.astype(np.float32) * 0.6).astype(np.uint8)
    scene[80:100, 90:110] = faded

    scene_file = tmp_path / "scene.png"
    tpl_file = tmp_path / "template.png"
    cv2.imwrite(str(scene_file), scene)
    cv2.imwrite(str(tpl_file), template)

    return str(scene_file), str(tpl_file)


def test_match_template_saves_all_default(tmp_path):
    scene_path, tpl_path = _make_test_images(tmp_path)

    # default should draw all kept matches (we expect two positions; at least one match saved)
    matched = match_template(scene_path, tpl_path, threshold=0.4, save_result=True)
    assert matched

    out_path = os.path.splitext(scene_path)[0] + "_matched_" + os.path.splitext(os.path.basename(tpl_path))[0] + os.path.splitext(scene_path)[1]
    assert os.path.isfile(out_path)


def test_match_template_draw_all_saves(tmp_path):
    scene_path, tpl_path = _make_test_images(tmp_path)

    # explicit draw_all True should act like the default and annotate all matches
    matched = match_template(scene_path, tpl_path, threshold=0.4, save_result=True, draw_all=True)
    assert matched

    out_path = os.path.splitext(scene_path)[0] + "_matched_" + os.path.splitext(os.path.basename(tpl_path))[0] + os.path.splitext(scene_path)[1]
    assert os.path.isfile(out_path)


def test_match_template_single_best_message(tmp_path, capsys):
    scene_path, tpl_path = _make_test_images(tmp_path)

    # Request single best: draw_all=False
    matched = match_template(scene_path, tpl_path, threshold=0.4, save_result=False, draw_all=False)
    assert matched
    captured = capsys.readouterr()
    # Should print message about a single best match
    assert "Detected 1 best match" in captured.out
