from collections import OrderedDict

from JobMetaData import JobMetaData


class ShockwaveScheduler(object):
    def __init__(
        self,
        ngpus: int,
        gram: int,
        init_metadata: OrderedDict,
        future_nrounds: int,
        round_duration: int,
        solver_preference: list,
        solver_rel_gap: float,
        solver_num_threads: int,
        solver_timeout: float,
        n_epoch_vars_max: int,
        logapx_bases: list,
        logapx_origin: dict,
        k: float,
        lam: float,
        rhomax: float,
    ):
        self.ngpus = ngpus
        self.gram = gram
        self.future_nrounds = future_nrounds
        self.round_duration = round_duration
        self.solver_preference = solver_preference
        self.solver_rel_gap = solver_rel_gap
        self.solver_num_threads = solver_num_threads
        self.solver_timeout = solver_timeout
        self.n_epoch_vars_max = n_epoch_vars_max
        self.logapx_bases = logapx_bases
        self.logapx_origin = logapx_origin
        self.k = k
        self.lam = lam
        self.rhomax = rhomax
        self.metadata = OrderedDict()
        self.round_ptr = 0
        self.resolve = True
        self.completed_jobs = OrderedDict()
        for jobid, jobobj in init_metadata.items():
            self.add_metadata(jobid, jobobj)

    # Public interface: ShockwaveScheduler.round_schedule
    def round_schedule(self):
        """Return the active job IDs selected for the current round."""
        raise NotImplementedError("scheduler policy removed")

    def increment_round_ptr(self):
        self.round_ptr += 1

    def set_resolve(self):
        self.resolve = True

    def clear_resolve(self):
        self.resolve = False

    def schedule_progress(self, jobid, epoch_progress, share_update=True):
        job = self.metadata[jobid]
        job.set_epoch_progress(epoch_progress)
        job.reset_waiting_delay()

    def deschedule_waiting_delay(self, jobid, delay):
        if jobid in self.metadata:
            self.metadata[jobid].add_waiting_delay(delay)

    def add_metadata(self, jobid, jobobj: JobMetaData, share_update=True):
        if jobid in self.metadata:
            raise ValueError("duplicate job ID")
        self.metadata[jobid] = jobobj
        self.set_resolve()

    def remove_metadata(self, jobid, share_update=True):
        self.completed_jobs[jobid] = self.metadata.pop(jobid)
        self.set_resolve()
