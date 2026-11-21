FROM verifier-base AS common
ARG DEBIAN_FRONTEND=noninteractive

# The calibration's history (`reference/history/`: earlier measurement sources and
# calibrations, with the final's rows and its seed), which nothing reads, is removed in a
# stage of its own, from which the final judge's image copies the rest, so that it is in none
# of its layers.
FROM common AS judge-eval
COPY judge /judge
RUN rm --recursive --force /judge/reference/history

# The final seed is for the final evaluation alone, and the calibration is bound to the
# measurement's exact sources. The train judge's `measurement.py` (`train/`) has no final
# seed; `rebind.py` then keeps the train workloads in the calibration and binds it to that
# measurement. All of it happens in a stage of its own, from which the train judge's image
# copies the judge, so that the final seed and the final workloads' calibration are in none of
# its layers.
FROM common AS judge-train
COPY judge /judge
COPY train/ /judge/
RUN rm --recursive --force /judge/reference/history
COPY rebind.py /rebind.py
RUN python3 /rebind.py /judge \
    && rm /rebind.py
RUN ! grep --recursive --quiet --fixed-strings FINAL_SEED /judge

# The train judge's image: the judge with no final seed, and the calibration of the train
# workloads alone, re-bound to the measurement.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge as it is, without the calibration's history.
FROM common AS eval
COPY --chmod=a+rX,a-w --from=judge-eval /judge /opt/frontier/task
USER 65532:65532
