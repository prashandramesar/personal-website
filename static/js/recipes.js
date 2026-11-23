// Behaviour for the private recipe pages. Each part only runs on the pages that
// have its elements, and every page still works without this script.
(function () {
    // Recipe page: show or hide the amounts as the source wrote them.
    function setUpOriginalsButton() {
        var button = document.getElementById('originals-button');
        if (!button) {
            return;
        }
        var originals = document.querySelectorAll('[data-original]');
        var showing = false;
        button.addEventListener('click', function () {
            showing = !showing;
            originals.forEach(function (original) {
                original.classList.toggle('hidden', !showing);
                original.classList.toggle('block', showing);
            });
            button.setAttribute('aria-expanded', showing ? 'true' : 'false');
            button.textContent = showing ? 'Hide original amounts' : 'Show original amounts';
        });
        button.classList.remove('hidden');
    }

    // Recipe page: keep the phone's screen on while cooking.
    function setUpKeepScreenOn() {
        var control = document.getElementById('screen-control');
        if (!control || !('wakeLock' in navigator)) {
            return;
        }
        var button = document.getElementById('screen-button');
        var message = document.getElementById('screen-message');
        var lock = null;
        var wanted = false;
        var requesting = false;

        function show(isOn) {
            button.setAttribute('aria-pressed', isOn ? 'true' : 'false');
            button.textContent = isOn ? 'Screen stays on: tap to turn off' : 'Keep screen on';
        }

        function showMessage(text) {
            message.textContent = text;
            message.classList.toggle('hidden', !text);
        }

        function stopTrying(text) {
            wanted = false;
            show(false);
            showMessage(text);
        }

        function onRelease() {
            lock = null;
            // Leaving the page releases the lock, and coming back asks again.
            // If it's released while visible, the phone decided to turn it off.
            if (wanted && document.visibilityState === 'visible') {
                stopTrying('Your phone turned this off. Tap to try again.');
            }
        }

        function requestLock() {
            if (requesting || lock) {
                return;
            }
            requesting = true;
            navigator.wakeLock.request('screen').then(function (newLock) {
                requesting = false;
                // It may have been turned off while the request was in progress.
                if (!wanted) {
                    newLock.release();
                    return;
                }
                lock = newLock;
                lock.addEventListener('release', onRelease);
                show(true);
                showMessage('');
            }).catch(function () {
                requesting = false;
                stopTrying("Your phone wouldn't keep the screen on. Battery saver can block this.");
            });
        }

        button.addEventListener('click', function () {
            if (wanted) {
                wanted = false;
                show(false);
                showMessage('');
                if (lock) {
                    lock.release();
                }
                return;
            }
            wanted = true;
            requestLock();
        });

        document.addEventListener('visibilitychange', function () {
            if (wanted && document.visibilityState === 'visible') {
                requestLock();
            }
        });

        control.classList.remove('hidden');
    }

    // Review and edit pages: add, remove and correct ingredient and step lines.
    function setUpRecipeForm() {
        // Changing a line means you've checked it, so its notes no longer apply.
        document.addEventListener('input', function (event) {
            if (!event.target.hasAttribute('data-clears-notes')) {
                return;
            }
            var line = event.target.closest('[data-line]');
            line.querySelector('[data-line-flags]').value = '';
            line.querySelectorAll('[data-line-note]').forEach(function (note) {
                note.remove();
            });
        });

        // A removed line leaves the page, so none of its fields are sent.
        document.addEventListener('click', function (event) {
            var button = event.target.closest('[data-remove-line]');
            if (button) {
                button.closest('[data-line]').remove();
            }
        });

        document.querySelectorAll('[data-add-line]').forEach(function (button) {
            var kind = button.getAttribute('data-add-line');
            var list = document.querySelector('[data-lines="' + kind + '"]');
            var template = document.getElementById('new-' + kind);
            button.addEventListener('click', function () {
                list.appendChild(template.content.cloneNode(true));
                var lines = list.querySelectorAll('[data-line]');
                lines[lines.length - 1].querySelector('input, textarea').focus();
                showRemoveButtons();
            });
            button.classList.remove('hidden');
        });

        function showRemoveButtons() {
            document.querySelectorAll('[data-remove-line]').forEach(function (button) {
                button.classList.remove('hidden');
            });
        }
        showRemoveButtons();
    }

    setUpOriginalsButton();
    setUpKeepScreenOn();
    setUpRecipeForm();
})();
