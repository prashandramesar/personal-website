// Behaviour shared by every page: the theme toggle and the mobile menu.
(function () {
    var root = document.documentElement;
    var DARK_BAR = '#0e1116';
    var LIGHT_BAR = '#fbfcfd';

    function currentTheme() {
        var chosen = root.getAttribute('data-theme');
        if (chosen) {
            return chosen;
        }
        return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }

    function updateBrowserBar() {
        // An explicit choice can differ from the system setting these tags assume.
        var color = currentTheme() === 'dark' ? DARK_BAR : LIGHT_BAR;
        document.querySelectorAll('meta[name="theme-color"]').forEach(function (tag) {
            tag.setAttribute('content', color);
        });
    }

    function updateToggleLabel(button) {
        var next = currentTheme() === 'dark' ? 'light' : 'dark';
        button.setAttribute('aria-label', 'Switch to ' + next + ' mode');
    }

    var themeButton = document.getElementById('theme-toggle');
    if (themeButton) {
        updateToggleLabel(themeButton);
        themeButton.addEventListener('click', function () {
            var next = currentTheme() === 'dark' ? 'light' : 'dark';
            root.setAttribute('data-theme', next);
            try {
                localStorage.setItem('theme', next);
            } catch (e) {
                // Blocked storage only means the choice isn't remembered.
            }
            updateToggleLabel(themeButton);
            updateBrowserBar();
        });
    }

    if (root.getAttribute('data-theme')) {
        updateBrowserBar();
    }

    var menuButton = document.getElementById('menu-toggle');
    var menu = document.getElementById('mobile-menu');
    if (menuButton && menu) {
        var setMenu = function (open) {
            menu.setAttribute('data-open', open ? 'true' : 'false');
            menuButton.setAttribute('aria-expanded', open ? 'true' : 'false');
            menuButton.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
        };
        menuButton.addEventListener('click', function () {
            setMenu(menu.getAttribute('data-open') !== 'true');
        });
        document.addEventListener('keydown', function (event) {
            if (event.key === 'Escape' && menu.getAttribute('data-open') === 'true') {
                setMenu(false);
                menuButton.focus();
            }
        });
    }

    // Forms with data-busy show progress, and can't be submitted twice.
    document.addEventListener('submit', function (event) {
        var form = event.target;
        if (event.defaultPrevented || !form.hasAttribute('data-busy')) {
            return;
        }
        var button = form.querySelector('button[type="submit"]');
        if (button) {
            button.setAttribute('data-idle-label', button.textContent);
            button.textContent = form.getAttribute('data-busy');
            button.disabled = true;
        }
    });

    // Going back restores a page as it was left, including a disabled button.
    window.addEventListener('pageshow', function (event) {
        if (!event.persisted) {
            return;
        }
        document.querySelectorAll('button[data-idle-label]').forEach(function (button) {
            button.textContent = button.getAttribute('data-idle-label');
            button.disabled = false;
        });
    });

    function showToast(text) {
        var old = document.getElementById('toast');
        if (old) {
            old.remove();
        }
        var toast = document.createElement('p');
        toast.id = 'toast';
        toast.className = 'toast';
        toast.setAttribute('role', 'alert');
        toast.textContent = text;
        document.body.appendChild(toast);
        setTimeout(function () {
            toast.remove();
        }, 5000);
    }

    // A star changes at once and saves in the background. Without this script,
    // the form still works as a normal page submission.
    function setStar(form, starred) {
        var button = form.querySelector('button');
        form.querySelector('input[name="favorite"]').value = starred ? '0' : '1';
        button.setAttribute('aria-pressed', starred ? 'true' : 'false');
        var label = button.getAttribute(starred ? 'data-label-on' : 'data-label-off');
        button.setAttribute('aria-label', label || (starred ? 'Remove from favorites' : 'Add to favorites'));
    }

    document.querySelectorAll('[data-star-form]').forEach(function (form) {
        var saving = false;
        form.addEventListener('submit', function (event) {
            event.preventDefault();
            if (saving) {
                return;
            }
            saving = true;
            var body = new URLSearchParams(new FormData(form));
            var starred = body.get('favorite') === '1';
            var row = form.closest('[data-favorites-list] > li');
            setStar(form, starred);
            if (row && !starred) {
                row.classList.add('hidden');
                if (!row.parentElement.querySelector('li:not(.hidden)')) {
                    row.parentElement.classList.add('hidden');
                    document.getElementById('favorites-empty').classList.remove('hidden');
                }
            }
            fetch(form.action, { method: 'POST', body: body, credentials: 'same-origin' })
                .then(function (response) {
                    // An expired session redirects to the login page, which loads fine.
                    if (!response.ok || response.url.indexOf('/recipes/login') !== -1) {
                        throw new Error('not saved');
                    }
                })
                .catch(function () {
                    setStar(form, !starred);
                    if (row) {
                        row.classList.remove('hidden');
                        row.parentElement.classList.remove('hidden');
                        document.getElementById('favorites-empty').classList.add('hidden');
                    }
                    showToast("That star wasn't saved. Check your connection, or log in again.");
                })
                .finally(function () {
                    saving = false;
                });
        });
    });
})();
