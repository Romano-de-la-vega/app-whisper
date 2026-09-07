"""Native desktop composition; services own transcription and audio processing."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QAction, QCloseEvent, QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..core.config import LANGS, MODEL_APPROX_SIZE, MODELS_CLOUD, MODELS_LOCAL
from ..core.paths import AppPaths
from ..core.settings import SettingsStore
from ..models.transcription import TranscriptionOptions
from .theme import apply_theme
from .widgets.file_list import FileList
from .widgets.recording_panel import RecordingPanel
from .widgets.results_panel import ResultsPanel
from .workers import TaskRunner

OUTPUTS = {
    'transcription': 'Transcription uniquement',
    'resume': 'Résumé',
    'compte_rendu': 'Compte rendu',
    'note_de_cadrage': 'Note de cadrage',
    'cahier_des_charges': 'Cahier des charges',
    'procedure_technique': 'Procédure / documentation technique',
    'rapport_analyse': "Rapport d’analyse / étude de faisabilité",
    'support_formation': 'Support de formation / guide utilisateur',
}
STATUSES = {
    'pending': 'En attente', 'queued': 'En file d’attente', 'running': 'Traitement en cours',
    'cancelling': 'Annulation en cours…', 'done': 'Terminé', 'partial': 'Terminé avec des erreurs — résultats partiels conservés',
    'error': 'Échec — consultez le journal', 'cancelled': 'Annulé — résultats disponibles conservés',
}
TERMINAL = {'done', 'partial', 'error', 'cancelled'}


class MainWindow(QMainWindow):
    def __init__(
        self, paths: AppPaths, *, jobs: Any = None, media: Any = None, models: Any = None,
        devices: Any = None, recording: Any = None, monitor: Any = None, settings: Any = None,
    ) -> None:
        super().__init__()
        from ..services.audio_device_service import AudioDeviceService
        from ..services.job_service import JobService
        from ..services.media_service import MediaService
        from ..services.model_service import ModelService
        from ..services.recording_service import RecordingService

        self.paths = paths
        self.settings = settings if settings is not None else SettingsStore(paths)
        self.jobs = jobs if jobs is not None else JobService(paths, load_history=False)
        self.media = media if media is not None else MediaService(paths)
        self.models = models if models is not None else ModelService(paths)
        self.devices = devices if devices is not None else AudioDeviceService()
        self.recording = recording if recording is not None else RecordingService(paths, self.devices)
        self.runner = TaskRunner(self)
        self.runner.error.connect(self.show_notice)
        self._active = False
        self._initializing = True
        self._job_id: str | None = None
        self._cancel_requested = False
        self._poll_pending = False
        self._closing = False
        self._close_ready = False
        self._recorder_closed = False
        self._recorder_close_started = False
        self._monitor_finished = monitor is False
        self._last_logs = ''
        self._restoring = True
        self.setWindowTitle(f'Transcripteur Whisper {__version__}')
        self.setWindowIcon(QIcon(str(paths.asset_path('icon.ico'))))
        self.resize(1040, 830)
        self.setMinimumSize(800, 630)
        self._build_ui()
        self._restore_preferences()
        self._restoring = False
        self._apply_theme()
        self._connect_settings()

        self.monitor = None
        if monitor is not False:
            if monitor is None:
                from ..audio.device_monitor import DeviceMonitor
                monitor = DeviceMonitor(self.devices, recording=self.recording, parent=self)
            self.monitor = monitor
            monitor.devices_changed.connect(self.recording_panel.update_devices)
            monitor.error.connect(self.show_notice)
            monitor.finished.connect(self._monitor_stopped)
            monitor.start()
        else:
            self.recording_panel.status.setText('Détection audio désactivée pour cette session de validation.')
        self.recording_panel.scan_recovery()
        history_loader = self.jobs.history if jobs is not None else self.jobs.load_history
        self.runner.submit(history_loader, self._load_history, self.show_notice, self._initialized)
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(400)
        self.poll_timer.timeout.connect(self._poll_job)
        self.close_timer = QTimer(self)
        self.close_timer.setInterval(100)
        self.close_timer.timeout.connect(self._complete_close)

    def _build_ui(self) -> None:
        self.logo = QLabel()
        self.logo.setMaximumSize(150, 55)
        title = QLabel('Transcripteur Whisper')
        title.setStyleSheet('font-size: 21pt; font-weight: bold;')
        subtitle = QLabel('Vos fichiers audio et vidéo, en texte. En local ou avec OpenAI.')
        subtitle.setWordWrap(True)
        header_text = QVBoxLayout()
        header_text.addWidget(title)
        header_text.addWidget(subtitle)
        header = QHBoxLayout()
        header.addWidget(self.logo)
        header.addLayout(header_text, 1)
        self.theme_button = QPushButton()
        self.theme_button.clicked.connect(self.toggle_theme)
        header.addWidget(self.theme_button)
        self.tabs = QTabWidget()
        self.mode = QComboBox()
        self.mode.addItem('Local (CPU int8)', 'local')
        self.mode.addItem('OpenAI API', 'api')
        self.model = QComboBox()
        self.language = QComboBox()
        self.language.addItem('Détection automatique', None)
        for label, code in LANGS.items():
            self.language.addItem(label, code)
        self.output_type = QComboBox()
        for value, label in OUTPUTS.items():
            self.output_type.addItem(label, value)
        self.output_name = QLineEdit()
        self.output_name.setMaxLength(80)
        self.output_name.setPlaceholderText('Ex. Réunion de lancement')
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText('Votre clé OpenAI — conservée uniquement en mémoire')
        self.api_key.setObjectName('apiKey')
        self.api_label = QLabel('Clé OpenAI')
        self.api_hint = QLabel('En mode API, les fichiers et textes sont envoyés à OpenAI. Des frais peuvent s’appliquer à votre compte.')
        self.api_hint.setWordWrap(True)
        self.options = QGroupBox('Paramètres')
        form = QFormLayout(self.options)
        form.addRow('Mode', self.mode)
        form.addRow('Modèle', self.model)
        form.addRow('Langue', self.language)
        form.addRow('Sortie', self.output_type)
        form.addRow('Nom du résultat', self.output_name)
        form.addRow(self.api_label, self.api_key)
        form.addRow(self.api_hint)
        self.files = FileList(self.runner, self.media)
        self.files.notice.connect(self.show_notice)
        self.files.changed.connect(self._update_start_enabled)
        transcribe = QWidget()
        transcribe_layout = QVBoxLayout(transcribe)
        transcribe_layout.addWidget(self.options)
        transcribe_layout.addWidget(self.files, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(transcribe)
        self.tabs.addTab(scroll, 'Transcrire des fichiers')
        self.recording_panel = RecordingPanel(self.runner, self.recording, self.settings)
        self.recording_panel.recording_saved.connect(self._recording_saved)
        self.recording_panel.notice.connect(self.show_notice)
        self.recording_panel.state_changed.connect(self._update_start_enabled)
        self.recording_panel.shutdown_failed.connect(self._shutdown_error)
        recording_tab = QWidget()
        recording_layout = QVBoxLayout(recording_tab)
        recording_layout.addWidget(self.recording_panel)
        recording_layout.addStretch()
        self.tabs.addTab(recording_tab, 'Enregistrer le son')
        self.results = ResultsPanel(self.runner, self.jobs, self.paths.results)
        self.results.notice.connect(self.show_notice)
        self.history = QComboBox()
        self.history.setAccessibleName('Historique des traitements')
        self.history.currentIndexChanged.connect(self._history_selected)
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        results_layout.addWidget(self.history)
        results_layout.addWidget(self.results)
        self.tabs.addTab(results_tab, 'Résultats')
        self.logs = QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMaximumBlockCount(1000)
        self.tabs.addTab(self.logs, 'Journal')
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat('%p %')
        self.state = QLabel('Prêt')
        self.state.setWordWrap(True)
        self.start_button = QPushButton('Démarrer la transcription')
        self.start_button.setObjectName('startButton')
        self.start_button.clicked.connect(self.start_transcription)
        self.cancel_button = QPushButton('Annuler')
        self.cancel_button.clicked.connect(self.cancel_transcription)
        self.cancel_button.setEnabled(False)
        actions = QHBoxLayout()
        actions.addWidget(self.start_button)
        actions.addWidget(self.cancel_button)
        actions.addStretch()
        self.notice = QLabel('')
        self.notice.setWordWrap(True)
        self.notice.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.addLayout(header)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.progress)
        layout.addWidget(self.state)
        layout.addLayout(actions)
        layout.addWidget(self.notice)
        self.setCentralWidget(central)
        about = QAction('À propos', self)
        about.triggered.connect(self.show_about)
        self.menuBar().addMenu('Aide').addAction(about)
        self.statusBar().showMessage('Fait par Romain Cieslik · Whisper + VAD Silero')
        self._update_start_enabled()

    def _restore_preferences(self) -> None:
        self.mode.setCurrentIndex(max(0, self.mode.findData(self.settings.get('mode', 'local'))))
        self._mode_changed()
        self.language.setCurrentIndex(max(0, self.language.findData(self.settings.get('language', 'fr'))))
        if self.mode.currentData() == 'api':
            self.output_type.setCurrentIndex(max(0, self.output_type.findData(self.settings.get('output_type', 'transcription'))))

    def _connect_settings(self) -> None:
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.model.currentIndexChanged.connect(self._save_preferences)
        self.language.currentIndexChanged.connect(self._save_preferences)
        self.output_type.currentIndexChanged.connect(self._save_preferences)

    def _mode_changed(self) -> None:
        api = self.mode.currentData() == 'api'
        self.model.blockSignals(True)
        self.model.clear()
        for label, value in ((item, item) for item in MODELS_CLOUD) if api else MODELS_LOCAL.items():
            self.model.addItem(label, value)
        preferred = self.settings.get('api_model' if api else 'local_model', self.model.itemData(0))
        self.model.setCurrentIndex(max(0, self.model.findData(preferred)))
        self.model.blockSignals(False)
        self.api_key.setVisible(api)
        self.api_label.setVisible(api)
        self.api_hint.setVisible(api)
        self.output_type.setEnabled(api)
        if not api:
            self.output_type.setCurrentIndex(0)
        self._save_preferences()

    def _save_preferences(self) -> None:
        if self._restoring:
            return
        values = {
            'mode': self.mode.currentData(),
            'api_model' if self.mode.currentData() == 'api' else 'local_model': self.model.currentData(),
            'language': self.language.currentData(), 'output_type': self.output_type.currentData(),
        }
        try:
            for key, value in values.items():
                self.settings.set(key, value)
        except OSError:
            self.show_notice('Les préférences ne peuvent pas être enregistrées dans votre dossier utilisateur.')

    def _apply_theme(self) -> None:
        dark = self.settings.get('theme', 'dark') == 'dark'
        apply_theme(dark)
        pixmap = QPixmap(str(self.paths.asset_path('logo_white.png' if dark else 'logo.png')))
        self.logo.setPixmap(pixmap.scaled(140, 48, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.theme_button.setText('Mode clair' if dark else 'Mode sombre')

    def toggle_theme(self) -> None:
        self.settings.set('theme', 'light' if self.settings.get('theme', 'dark') == 'dark' else 'dark')
        self._apply_theme()

    def show_about(self) -> None:
        QMessageBox.about(self, 'À propos de Transcripteur Whisper',
                          f'Transcripteur Whisper {__version__}\nApplication Windows native · PySide6 / Qt 6\n'
                          'Whisper local ou OpenAI · Microphone et son du PC\nCréé par Romain Cieslik')

    @Slot(str)
    def show_notice(self, message: str) -> None:
        key = self.api_key.text() if hasattr(self, 'api_key') else ''
        if key:
            message = message.replace(key, '[clé masquée]')
        self.notice.setText(message)
        self.statusBar().showMessage(message, 12000)

    def _update_start_enabled(self) -> None:
        if hasattr(self, 'start_button'):
            self.start_button.setEnabled(bool(self.files.paths) and not self._active
                                         and not self.recording_panel.busy and not self._closing and not self._initializing)

    def _initialized(self) -> None:
        self._initializing = False
        self._update_start_enabled()

    def _set_active(self, active: bool) -> None:
        self._active = active
        self.options.setEnabled(not active)
        self.files.setEnabled(not active)
        self.recording_panel.setEnabled(not active)
        self.cancel_button.setEnabled(active and not self._cancel_requested)
        self._update_start_enabled()

    def start_transcription(self) -> None:
        if self._initializing or self._active or not self.files.paths or self.recording_panel.busy:
            return
        options = TranscriptionOptions(mode=self.mode.currentData(), model=self.model.currentData(),
                                       language=self.language.currentData(), output_type=self.output_type.currentData(),
                                       output_name=self.output_name.text().strip())
        if options.mode == 'api' and not self.api_key.text().strip():
            self.show_notice('Saisissez votre clé OpenAI pour utiliser le mode API.')
            self.api_key.setFocus()
            return
        self._cancel_requested = False
        self._job_id = None
        self._set_active(True)
        self.progress.setValue(0)
        self.state.setText('Vérification du modèle…' if options.mode == 'local' else 'Préparation…')
        self.show_notice('')
        if options.mode == 'local':
            self.runner.submit(lambda: self.models.available(options.model),
                               lambda available: self._model_checked(options, available), self._launch_failed)
        else:
            self._submit_job(options)

    def _model_checked(self, options: TranscriptionOptions, available: bool) -> None:
        if self._cancel_requested or self._closing:
            self._set_active(False)
            self.state.setText('Annulé')
            return
        if not available:
            size = MODEL_APPROX_SIZE.get(options.model)
            estimate = f'Environ {size / 1024 ** 3:.2f} Go' if size else 'La taille dépend du modèle'
            response = QMessageBox.question(
                self, 'Télécharger le modèle Whisper',
                f'Le modèle « {self.model.currentText()} » doit être téléchargé.\n{estimate}. '
                'Une connexion Internet est nécessaire.\nIl sera conservé pour vos prochains lancements.\n\n'
                'Télécharger le modèle et démarrer la transcription ?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
            )
            if response != QMessageBox.StandardButton.Yes:
                self._set_active(False)
                self.state.setText('Téléchargement non lancé')
                return
        self._submit_job(options)

    def _submit_job(self, options: TranscriptionOptions) -> None:
        if self._cancel_requested or self._closing:
            self._set_active(False)
            return
        files = self.files.paths
        key = self.api_key.text().strip() if options.mode == 'api' else ''
        self.runner.submit(lambda: self.jobs.submit(files, options, api_key=key), self._job_submitted, self._launch_failed)

    def _job_submitted(self, identifier: str) -> None:
        self._job_id = identifier
        self._last_logs = ''
        if self._cancel_requested or self._closing:
            self.runner.submit(lambda: self.jobs.cancel(identifier), failure=self.show_notice)
        self.poll_timer.start()
        self._poll_job()

    def _launch_failed(self, message: str) -> None:
        self._set_active(False)
        self.state.setText('Échec du lancement')
        self.show_notice(message)

    def cancel_transcription(self) -> None:
        self._cancel_requested = True
        self.cancel_button.setEnabled(False)
        self.state.setText('Annulation en cours… Les résultats déjà produits seront conservés.')
        if self._job_id:
            identifier = self._job_id
            self.runner.submit(lambda: self.jobs.cancel(identifier), failure=self.show_notice)

    def _poll_job(self) -> None:
        if self._poll_pending or not self._job_id:
            return
        identifier = self._job_id
        self._poll_pending = True
        self.runner.submit(lambda: self.jobs.snapshot(identifier), self._render_job, self.show_notice,
                           lambda: setattr(self, '_poll_pending', False))

    def _render_job(self, snapshot: dict) -> None:
        status = snapshot.get('status', 'pending')
        self.state.setText(STATUSES.get(status, status))
        download = snapshot.get('model_download_progress')
        if snapshot.get('mode') == 'local' and download is not None and status not in TERMINAL and 0 < float(download) < 1:
            self.progress.setValue(round(float(download) * 1000))
            self.progress.setFormat('Téléchargement du modèle : %p %')
        else:
            self.progress.setFormat('%p %')
            self.progress.setValue(round(float(snapshot.get('progress') or 0) * 1000))
        self.files.update_progress(snapshot.get('files', []))
        logs = '\n'.join(snapshot.get('logs', []))
        if logs != self._last_logs:
            self._last_logs = logs
            self.logs.setPlainText(logs)
            self.logs.verticalScrollBar().setValue(self.logs.verticalScrollBar().maximum())
        self.results.load(self._job_id, snapshot)
        if status in TERMINAL:
            self.poll_timer.stop()
            self._set_active(False)
            self.runner.submit(self.jobs.history, self._load_history, self.show_notice)
            if self.results.entries and not self._closing:
                self.tabs.setCurrentIndex(2)

    def _load_history(self, history: list[dict]) -> None:
        selected = self._job_id or self.history.currentData()
        self.history.blockSignals(True)
        self.history.clear()
        for item in history:
            identifier = item.get('id') or item.get('job_id')
            files = item.get('files') or []
            source_name = files[0].get('name') if files else None
            label = item.get('output_name') or source_name or str(identifier)[:12]
            if len(files) > 1:
                label += f' (+ {len(files) - 1} fichier(s))'
            if item.get('created_at'):
                try:
                    date = datetime.fromisoformat(item['created_at']).astimezone().strftime('%d/%m/%Y %H:%M')
                    label = f'{date} · {label}'
                except ValueError:
                    pass
            self.history.addItem(f'{label} · {STATUSES.get(item.get("status"), item.get("status", ""))}', identifier)
        self.history.setCurrentIndex(max(0, self.history.findData(selected)))
        self.history.blockSignals(False)
        if not self._active:
            self._history_selected()

    def _history_selected(self) -> None:
        identifier = self.history.currentData()
        if identifier and not self._active:
            self.runner.submit(lambda: self.jobs.snapshot(identifier),
                               lambda snapshot: self.results.load(identifier, snapshot), self.show_notice)

    def _recording_saved(self, path: Path) -> None:
        self.files.add_paths([path])
        if not self._closing:
            self.tabs.setCurrentIndex(0)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._close_ready:
            self.api_key.clear()
            event.accept()
            return
        event.ignore()
        if self._closing:
            return
        self._closing = True
        self.centralWidget().setEnabled(False)
        self.state.setText('Fermeture : annulation des traitements et conservation de l’enregistrement…')
        self.cancel_transcription()
        self.recording_panel.shutdown()
        if self.monitor:
            self.monitor.stop()
        self.close_timer.start()

    def _monitor_stopped(self) -> None:
        self._monitor_finished = True

    def _complete_close(self) -> None:
        if self.recording_panel.busy or self.jobs.busy or self.runner.active_count or not self._monitor_finished:
            return
        if not self._recorder_close_started:
            self._recorder_close_started = True
            self.runner.submit(self._close_services,
                               lambda _: setattr(self, '_recorder_closed', True), self._shutdown_error)
            return
        if not self._recorder_closed:
            return
        self.poll_timer.stop()
        self.close_timer.stop()
        self._close_ready = True
        self.close()

    def _close_services(self) -> None:
        getattr(self.recording, 'close', lambda: None)()
        self.jobs.shutdown(wait=True)

    def _shutdown_error(self, message: str) -> None:
        self.close_timer.stop()
        self._closing = False
        self._recorder_close_started = False
        self.centralWidget().setEnabled(True)
        self.recording_panel.resume_after_shutdown_error()
        if self.monitor:
            self._monitor_finished = False
            self.monitor.start()
        self.show_notice(f'Fermeture suspendue : {message}. Les fichiers de récupération sont conservés. Réessayez d’arrêter l’enregistrement.')
        self._update_start_enabled()
