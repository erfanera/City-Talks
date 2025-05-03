import styled, { createGlobalStyle } from 'styled-components';

export const GlobalStyle = createGlobalStyle`
    body {
        font-family: 'Cascadia Code', monospace;
        background-color: #f5f5f5;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
`;

export const Container = styled.div`
    display: grid;
    grid-template-columns: 2fr 1fr;
    grid-template-rows: auto auto auto auto;
    gap: 20px;
    width: 100%;
    height: 100%;
    margin: 0;
    padding: 20px;
    box-sizing: border-box;
`;

export const HeaderContainer = styled.div`
    grid-column: span 2;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 0 20px 0;
    border-bottom: 1px solid #d0d0d0;
`;

export const HeaderLogo = styled.div`
    display: flex;
    align-items: center;
    gap: 10px;
`;

export const Header = styled.h1`
    font-size: 32px;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    background-color: #3e5caa;
    padding: 8px 16px;
    border-radius: 0;
`;

export const HeaderNav = styled.div`
    display: flex;
    gap: 10px;
`;

export const Tagline = styled.p`
    grid-column: span 2;
    font-size: 18px;
    color: #3e5caa;
    text-align: center;
    margin: 0 0 20px 0;
`;

export const MapContainer = styled.div`
    grid-column: 1 / 2;
    background-color: #e1e8f7;
    border: 2px solid #3e5caa;
    border-radius: 12px;
    height: 100%;
    min-height: 1000px;
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;
    overflow: hidden;





    /* Ensure the iframe takes up the full space */
    iframe {
        width: 100%;
        height: 100%;
        border: none;
    }
`;




export const StreetView = styled.div`
    grid-column: 2 / 3;
    background-color: #dfe3ea;
    border: 2px solid #3e5caa;
    border-radius: 12px;
    height: 100%;
    min-height: 1000px;
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;
`;

export const PromptContainer = styled.div`
    grid-column: span 2;
    padding: 20px;
    background-color: #ffffff;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    display: flex;
    flex-direction: column;
    gap: 20px;
`;

export const TextArea = styled.textarea`
    width: 100%;
    height: 80px;
    padding: 12px;
    border: 2px solid #3e5caa;
    border-radius: 10px;
    outline: none;
    font-size: 16px;
    font-family: 'Cascadia Code', monospace;
    resize: none;
`;

export const Button = styled.button`
    padding: 10px 20px;
    background-color: #3e5caa;
    color: #fff;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-size: 16px;
    font-weight: 600;
    font-family: 'Cascadia Code', monospace;
    transition: background 0.3s ease-in-out;
    
    &:hover {
        background-color: #2c4885;
    }
    
    &.amenity-score-button {
        align-self: center;
        min-width: 180px;
    }
`;

export const GenerateButton = styled(Button)`
    width: 100%;
    padding: 15px;
    font-size: 18px;
    text-transform: none;
    max-width: 300px;
    align-self: center;
`;

export const FilterContainer = styled.div`
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 15px;
    
    @media (min-width: 768px) {
        grid-template-columns: repeat(4, 1fr);
    }
    
    div {
        display: flex;
        align-items: center;
        gap: 10px;
    }
`;

export const FilterCheckbox = styled.input`
    width: 20px;
    height: 20px;
    cursor: pointer;
    accent-color: #3e5caa;
`;

export const FilterLabel = styled.label`
    font-size: 16px;
    color: #333;
    cursor: pointer;
`;

export const AnalyticsSection = styled.div`
    grid-column: 1 / 2;
    padding: 20px;
    background-color: #ffffff;
    border-radius: 15px;
    border: 2px solid #3e5caa;
    display: flex;
    flex-direction: column;
    gap: 20px;
`;

export const AnalyticsTitle = styled.h2`
    font-size: 24px;
    color: #3e5caa;
    margin: 0;
`;

export const BarGraph = styled.div`
    flex: 1;
    background-color: #ffcccb;
    border-radius: 8px;
    position: relative;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 150px;
    
    .graph-title {
        font-size: 24px;
        font-weight: bold;
        color: #333;
        border: 2px solid #333;
        padding: 10px 20px;
    }
`;

export const BuildingInfoSection = styled.div`
    grid-column: 2 / 3;
    padding: 20px;
    background-color: #e6eeff;
    border-radius: 15px;
    border: 2px solid #3e5caa;
    display: flex;
    flex-direction: column;
    gap: 15px;
`;

export const BuildingInfoTitle = styled.h2`
    font-size: 24px;
    color: #3e5caa;
    margin: 0 0 10px 0;
    text-align: center;
`;

export const PropertyDetail = styled.div`
    display: flex;
    align-items: center;
    gap: 15px;
    padding: 8px 0;
`;

export const PropertyIcon = styled.div`
    width: 40px;
    height: 40px;
    display: flex;
    justify-content: center;
    align-items: center;
    border: 1px solid #3e5caa;
    border-radius: 5px;
    padding: 5px;
`;

export const PropertyValue = styled.div`
    flex: 1;
    display: flex;
    justify-content: space-between;
    align-items: center;
    
    div:first-child {
        font-size: 16px;
        color: #333;
    }
    
    .output {
        background-color: #fff;
        padding: 5px 10px;
        border-radius: 5px;
        min-width: 80px;
        text-align: center;
        font-weight: bold;
    }
`;
