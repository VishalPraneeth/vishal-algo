from flask import Blueprint, render_template, session, redirect, url_for, request, jsonify
from database.auth_db import get_auth_token, get_api_key_for_tradingview
from database.settings_db import get_analyze_mode
from services.sector_engine import fetch_sector_wise_scope
from utils.session import check_session_validity
from utils.logging import get_logger
from openalgo import api

logger = get_logger(__name__)

sector_heatmap_bp = Blueprint('sector_heatmap_bp',__name__,url_prefix='/')

# ---------------------------------------
# UI ROUTE
# ---------------------------------------
@sector_heatmap_bp.route('/sector-heatmap')
@check_session_validity
def sector_heatmap_dashboard():
    """
    Renders the Sector Heatmap UI page
    """
    login_username = session['user']
    AUTH_TOKEN = get_auth_token(login_username)

    if AUTH_TOKEN is None:
        logger.warning(f"No auth token found for user {login_username}")
        return redirect(url_for('auth.logout'))

    broker = session.get('broker')
    if not broker:
        logger.error("Broker not set in session")
        return "Broker not set in session", 400

    return render_template('sector_heatmap.html')
